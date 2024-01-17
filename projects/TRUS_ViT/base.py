import torch
from src.utils.accumulators import DictConcatenation
from tqdm import tqdm
import os
import argparse
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from src.utils.checkpoints import save_checkpoint
import numpy as np
import random
import matplotlib.pyplot as plt
import yaml
import json
from torch.nn import functional as F
from torch.utils.data import DataLoader
from projects.TRUS_ViT.utils import (
    show_prob_histogram,
    show_reliability_diagram,
    convert_patchwise_to_corewise_dataframe,
    apply_temperature_calibration,
)


class BaseExperiment:
    def __init__(self, args):
        self.args = args

    def run(self):
        logging.info("Starting main.py")

        os.makedirs(self.args.exp_dir, exist_ok=True)
        os.symlink(
            self.args.ckpt_dir,
            os.path.join(self.args.exp_dir, "checkpoints"),
            target_is_directory=True,
        )

        # save config

        with open(os.path.join(self.args.exp_dir, "config.yaml"), "w") as f:
            yaml.dump(OmegaConf.to_container(self.args, resolve=True), f)

        logging.info("Saved config.yaml")

        # set up wandb
        # if self.args.debug:
            # wandb.init = lambda **kwargs: None
            #wandb.log = lambda x: print(x)

        wandb.init(
            dir=self.args.exp_dir,
            config=OmegaConf.to_container(self.args, resolve=True),
            **self.args.wandb,
        )

        train_ds, val_ds, test_ds = self.create_datasets(self.args)
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        torch.random.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        print("CUDA?", torch.cuda.is_available())

        model = self.create_model()
        if self.args.from_ckpt is not None:
            model.load_state_dict(
                torch.load(
                    os.path.join(f"{self.args.from_ckpt}.pth"),
                    map_location=self.args.device,
                )
            )
            print(f"Loaded model from epoch {self.args.from_ckpt}")
        model = model.to(self.args.device)

        optimizer, sched = self.create_optimizer(model)

        epoch_out = DictConcatenation()
        for epoch in range(1, self.args.num_epochs + 1):
            logging.info(f"Starting epoch {epoch}")

            metrics = {}
            train_metrics, train_df = self.train_epoch(
                model, self.train_loader, optimizer, epoch
            )
            metrics.update({f"train/{k}": v for k, v in train_metrics.items()})

            eval_metrics, eval_dfs = self.eval_epoch(
                model, self.val_loader, self.test_loader, epoch
            )
            metrics.update(eval_metrics)

            metrics["lr"] = optimizer.param_groups[0]["lr"]
            metrics["epoch"] = epoch
            epoch_out(metrics)
            wandb.log(metrics)

            os.makedirs(
                os.path.join(self.args.exp_dir, f"epoch_{epoch}"), exist_ok=True
            )
            train_df.to_csv(
                os.path.join(self.args.exp_dir, f"epoch_{epoch}", "train.csv")
            )
            for name, df in eval_dfs.items():
                df.to_csv(
                    os.path.join(self.args.exp_dir, f"epoch_{epoch}", f"{name}.csv")
                )

            epoch_out.compute("dataframe").to_csv(
                os.path.join(self.args.exp_dir, "metrics.csv")
            )

            if sched is not None:
                sched.step()

            if epoch % self.args.checkpoint_freq == 0:
                torch.save(
                    model.state_dict(),
                    f"checkpoints/epoch_{epoch}.pth",
                )

        # post best values
        df = epoch_out.compute("dataframe")
        print(df.head())
        best_ind = df["val/patch_auc"].values.argmax()
        best_metrics = {
            f"{column}_best": df[column].values[best_ind] for column in df.columns
        }
        wandb.log(best_metrics)
        wandb.finish()

    def train_epoch(self, model, train_loader, optimizer, epoch):
        model.train()
        acc = DictConcatenation()
        losses = []
        for i, batch in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        ):
            outputs = self.train_step(model, batch, optimizer, epoch)
            losses.append(outputs['loss'].detach().cpu().mean())
            acc(outputs)
        df = acc.compute("dataframe")
        print(df.columns)
        metrics = self.compute_metrics(df)

        wandb.log({
            "epoch": epoch,
            "train/loss": sum(losses) / len(losses),
        })

        return metrics, df

    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat = model(x)
        prob = F.softmax(y_hat, dim=1)
        pred = prob.argmax(dim=1)
        confidence = prob.max(dim=1).values
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **metadata,
        }

    def eval_epoch(self, model, val_loader, test_loader, epoch):
        logging.info("Evaluating on validation set")

        model.eval()
        acc = DictConcatenation()
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch} validation", leave=False)
            ):
                out = self.eval_step(model, batch)
                losses.append(out['loss'].detach().cpu().mean())
                acc(out)

        val_df = acc.compute("dataframe")

        wandb.log({
            "epoch": epoch,
            "val/loss": sum(losses) / len(losses),
        })

        logging.info("Evaluating on test set")
        acc = DictConcatenation()
        with torch.no_grad():
            for i, batch in enumerate(
                tqdm(test_loader, desc=f"Epoch {epoch} validation", leave=False)
            ):
                out = self.eval_step(model, batch)
                acc(out)

        test_df = acc.compute("dataframe")

        val_df, test_df = apply_temperature_calibration(val_df, test_df)

        logging.info("Computing metrics")
        metrics = {}

        # make directory for logs for this epoch
        os.makedirs(os.path.join(self.args.exp_dir, f"epoch_{epoch}"), exist_ok=True)

        # Get patchwise metrics
        metrics.update(
            {f"val/patch_{k}": v for k, v in self.compute_metrics(val_df).items()}
        )
        metrics.update(
            {f"test/patch_{k}": v for k, v in self.compute_metrics(test_df).items()}
        )

        # Get corewise metrics
        val_df_corewise = convert_patchwise_to_corewise_dataframe(val_df)
        test_df_corewise = convert_patchwise_to_corewise_dataframe(test_df)

        metrics.update(
            {
                f"val/core_{k}": v
                for k, v in self.compute_metrics(val_df_corewise).items()
            }
        )
        metrics.update(
            {
                f"test/core_{k}": v
                for k, v in self.compute_metrics(test_df_corewise).items()
            }
        )

        return metrics, {
            "val_patchwise": val_df,
            "test_patchwise": test_df,
            "val_corewise": val_df_corewise,
            "test_corewise": test_df_corewise,
            #"test_ood": ood_df,
        }

    def eval_step(self, model, batch):
        x, y, metadata = batch
        x = x.to(self.args.device)
        y = y.to(self.args.device)
        y_hat = model(x)
        prob = F.softmax(y_hat, dim=1)
        confidence = prob.max(dim=1).values
        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": confidence,
            **metadata,
        }

    def create_model(self):
        from src.modeling.registry import resnet10

        return resnet10().cuda()

    def create_optimizer(self, model):
        if self.args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=0.9,
            )
        elif self.args.optimizer == "novograd":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.weight_decay,
            )
        else:
            raise ValueError(f"Optimizer {self.args.optimizer} not supported.")

        if self.args.scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            sched = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)

        elif self.args.scheduler == "cosine_restarts":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

            sched = CosineAnnealingWarmRestarts(
                optimizer, T_0=self.args.num_epochs // 3, T_mult=1
            )

        elif self.args.scheduler == "cosine_warmup":
            from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
            sched = LinearWarmupCosineAnnealingLR(
                optimizer,
                warmup_epochs=5,
                max_epochs=self.args.num_epochs,
            )

        else:
            sched = None

        return optimizer, sched

    def compute_metrics(self, dataframe):
        from sklearn.metrics import roc_auc_score, recall_score
        #from src.utils.metrics import brier_score, expected_calibration_error

        auc = roc_auc_score(dataframe.y, dataframe.prob_1)
        sensitivity = recall_score(dataframe.y, dataframe.prob_1 > 0.5)
        specificity = recall_score(1 - dataframe.y, dataframe.prob_1 < 0.5)

        #probs = dataframe.prob_1.values
        #targets = dataframe.y.values
        #preds = (probs > 0.5).astype(int)
        #conf = np.max(np.stack([probs, 1 - probs], axis=1), axis=1).squeeze()

        #ece, _ = expected_calibration_error(preds, conf, targets, n_bins=20)

        #brier = brier_score(probs, targets)

        return {
            "auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity
        }

    def create_datasets(self, args):
        from src.data.exact.cohort_selection import (
            get_cores_for_patients,
            get_patient_splits, get_tmi23_patient_splits,
            get_centerwise_patient_splits,
            remove_benign_cores_from_positive_patients,
            remove_cores_below_threshold_involvement,
            undersample_benign,
            undersample_benign_as_kfold,
        )

        if self.args.cross_val:
            if self.args.kfold_centerwise:
                patient_splits = get_centerwise_patient_splits(fold=self.args.fold)
            else:
                patient_splits = get_patient_splits(fold=self.args.fold)
        else: 
            patient_splits = get_tmi23_patient_splits()
        
        core_splits = [get_cores_for_patients(patients) for patients in patient_splits]
        core_splits = [
             remove_benign_cores_from_positive_patients(cores) for cores in core_splits
        ]
        core_splits = [
            remove_cores_below_threshold_involvement(core, threshold_pct=40)
            for core in core_splits
        ]
        train_cores, val_cores, test_cores = core_splits
        train_cores = remove_benign_cores_from_positive_patients(train_cores)
        # test_cores = remove_benign_cores_from_positive_patients(test_cores)

        if self.args.benign_undersampling:
            if self.args.undersample_kfold:
                train_cores = undersample_benign_as_kfold(train_cores)[self.args.undersample_fold_idx]
            else:
                train_cores = undersample_benign(train_cores, 
                                                 #seed=self.args.seed,
                                                 benign_to_cancer_ratio=self.args.benign_to_cancer_ratio)

        from src.data.exact.dataset.rf_datasets import PatchesDataset, PatchViewConfig
        
        from src.data.exact.transforms import TransformV3
        from src.data.exact.transforms import TensorImageAugmentation
        from src.data.exact.transforms import UltrasoundArrayAugmentation

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
        
        #print("Train cores:", train_cores)
        #print("Val cores:", val_cores)
        print("Test cores:", test_cores)

        train_dataset = PatchesDataset(
            core_specifier_list=train_cores,
            patch_view_config=patch_view_cfg,
            transform=train_transform,
            target_transform=self._label_transform,
        )
        val_dataset = PatchesDataset(
            core_specifier_list=val_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=self._label_transform,
        )
        test_dataset = PatchesDataset(
            core_specifier_list=test_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=self._label_transform,
        )

        return train_dataset, val_dataset, test_dataset

    def _label_transform(self, label):
        return torch.tensor(label, dtype=torch.long)