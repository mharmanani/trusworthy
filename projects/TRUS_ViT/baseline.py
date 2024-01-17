import torch
from src.utils.accumulators import DictConcatenation
from tqdm import tqdm
import os
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="config", config_name="baseline", version_base="1.3")
def main(args: DictConfig):
    # set up exp dir
    os.makedirs(args.exp_dir, exist_ok=False)

    # save config
    import yaml

    with open(os.path.join(args.exp_dir, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(args, resolve=True), f)

    # set up wandb
    if args.debug:
        wandb.init = lambda **kwargs: None
        wandb.log = lambda x: print(x)

    wandb.init(
        dir=args.exp_dir,
        config=OmegaConf.to_container(args, resolve=True),
        **args.wandb,
    )

    train_ds, val_ds, test_ds = create_datasets(args)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    from src.modeling.registry import create_model

    model = create_model(args.model_name)
    model = model.to(args.device)

    optimizer, sched = create_optimizer(args, model)

    epoch_out = DictConcatenation()
    for epoch in range(1, args.num_epochs + 1):
        metrics = {}
        train_metrics, train_df = train_epoch(
            args, model, train_loader, optimizer, epoch
        )
        metrics.update({f"train_{k}": v for k, v in train_metrics.items()})
        val_metrics, val_df = eval_epoch(args, model, val_loader, epoch)
        metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        test_metrics, test_df = eval_epoch(args, model, test_loader, epoch)
        metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        epoch_out(metrics)
        wandb.log(metrics)

        os.makedirs(os.path.join(args.exp_dir, f"epoch_{epoch}"), exist_ok=True)
        train_df.to_csv(os.path.join(args.exp_dir, f"epoch_{epoch}", "train.csv"))
        val_df.to_csv(os.path.join(args.exp_dir, f"epoch_{epoch}", "val.csv"))
        test_df.to_csv(os.path.join(args.exp_dir, f"epoch_{epoch}", "test.csv"))

        epoch_out.compute("dataframe").to_csv(os.path.join(args.exp_dir, "metrics.csv"))

        if sched is not None:
            sched.step()

    # post best values
    df = epoch_out.compute("dataframe")
    best_ind = df["val_patch_auc"].values.argmax()
    best_metrics = {
        f"{column}_best": df[column].values[best_ind] for column in df.columns
    }
    wandb.log(best_metrics)
    wandb.finish()


def train_epoch(args, model, train_loader, optimizer, epoch):
    model.train()
    acc = DictConcatenation()
    for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
        optimizer.zero_grad()
        x, y, metadata = batch
        x = x.to(args.device)
        y = y.to(args.device)
        y_hat = model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        acc(
            {
                "loss": torch.nn.functional.cross_entropy(y_hat, y, reduction="none"),
                "y": y,
                "prob": torch.nn.functional.softmax(y_hat, dim=1),
                **metadata,
            }
        )

    df = acc.compute("dataframe")
    metrics = compute_metrics(df)

    return metrics, df


def eval_epoch(args, model, val_loader, epoch):
    model.eval()
    acc = DictConcatenation()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch}", leave=False)):
            x, y, metadata = batch
            x = x.to(args.device)
            y = y.to(args.device)
            y_hat = model(x)
            acc(
                {
                    "loss": torch.nn.functional.cross_entropy(
                        y_hat, y, reduction="none"
                    ),
                    "y": y,
                    "prob": torch.nn.functional.softmax(y_hat, dim=1),
                    **metadata,
                }
            )

    df = acc.compute("dataframe")
    metrics = compute_metrics(df)

    return metrics, df


def compute_metrics(dataframe):
    from sklearn.metrics import roc_auc_score

    patch_auc = roc_auc_score(dataframe.y, dataframe.prob_1)
    core_prob = dataframe.groupby("core_specifier").prob_1.mean()
    core_y = dataframe.groupby("core_specifier").y.mean()
    core_auc = roc_auc_score(core_y, core_prob)

    return {
        "patch_auc": patch_auc,
        "core_auc": core_auc,
    }


def _label_transform(label):
    return torch.tensor(label, dtype=torch.long)


def create_datasets(args):
    from src.data.exact.cohort_selection import (
        get_cores_for_patients,
        get_patient_splits,
        remove_benign_cores_from_positive_patients,
        remove_cores_below_threshold_involvement,
        undersample_benign,
    )

    patient_splits = get_patient_splits(fold=args.fold)
    core_splits = [get_cores_for_patients(patients) for patients in patient_splits]
    # core_splits = [
    #     remove_benign_cores_from_positive_patients(cores) for cores in core_splits
    # ]
    core_splits = [
        remove_cores_below_threshold_involvement(core, threshold_pct=40)
        for core in core_splits
    ]
    train_cores, val_cores, test_cores = core_splits
    train_cores = remove_benign_cores_from_positive_patients(train_cores)
    train_cores = undersample_benign(
        train_cores,
    )

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew, PatchViewConfig
    from src.data.exact.transforms import (
        TransformV3,
        TransformA224, 
        TransformA192,
        TransformA100,
        TensorImageAugmentation,
        UltrasoundArrayAugmentation,
    )

    patch_view_cfg = PatchViewConfig(
        needle_region_only=True,
        prostate_region_only=False,
    )

    eval_transform = TransformV3()
    if args.augmentations_mode == "none":
        train_transform = eval_transform
    elif args.augmentations_mode == "tensor_augs":
        train_transform = TransformV3(
            tensor_transform=TensorImageAugmentation(
                random_resized_crop=True,
                random_affine_rotation=10,
                random_affine_translation=[0.1, 0.1],
            ),
        )
    elif args.augmentations_mode == "ultrasound_augs":
        train_transform = TransformV3(
            us_augmentation=UltrasoundArrayAugmentation(
                random_phase_shift=True,
                random_phase_distort=True,
                random_envelope_distort=True,
            )
        )
    elif args.augmentations_mode == "both":
        train_transform = TransformV3(
            tensor_transform=TensorImageAugmentation(
                random_resized_crop=True,
                random_affine_rotation=10,
                random_affine_translation=[0.1, 0.1],
            ),
            us_augmentation=UltrasoundArrayAugmentation(
                random_phase_shift=True,
                random_phase_distort=True,
                random_envelope_distort=True,
            ),
        )
    else:
        raise ValueError("Unknown augmentations_mode")

    train_dataset = PatchesDatasetNew(
        core_specifier_list=train_cores,
        patch_view_config=patch_view_cfg,
        patch_transform=train_transform,
        label_transform=_label_transform,
    )
    val_dataset = PatchesDatasetNew(
        core_specifier_list=val_cores,
        patch_view_config=patch_view_cfg,
        patch_transform=eval_transform,
        label_transform=_label_transform,
    )
    test_dataset = PatchesDatasetNew(
        core_specifier_list=test_cores,
        patch_view_config=patch_view_cfg,
        patch_transform=eval_transform,
        label_transform=_label_transform,
    )

    return train_dataset, val_dataset, test_dataset


def create_optimizer(args, model):
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=0.9,
        )
    elif args.optimizer == "novograd":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported.")

    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        sched = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    else:
        sched = None

    return optimizer, sched


if __name__ == "__main__":
    main()