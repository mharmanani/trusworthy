import torch
import torch.functional as F
from torch import nn
from projects.TRUS_ViT.losses import vicreg_loss_func
from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment
from projects.TRUS_ViT.base import BaseExperiment

from sklearn.linear_model import LogisticRegression

from vit_pytorch import ViT
from vit_pytorch.cct import CCT

from src.data.exact.transforms import TransformV3
from src.data.exact.transforms import TensorImageAugmentation
from src.data.exact.transforms import UltrasoundArrayAugmentation

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
from torch.utils.data import DataLoader
from projects.TRUS_ViT.utils import (
    show_prob_histogram,
    show_reliability_diagram,
    convert_patchwise_to_corewise_dataframe,
    apply_temperature_calibration,
)

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

class VICRegExperiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)
    
    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        x1, x2, y, metadata = batch
        x1 = x1.to(self.args.device)
        x2 = x2.to(self.args.device)
        loss = model.forward(x1, x2)
        loss.backward()
        optimizer.step()

        return {
            "loss": loss * torch.ones_like(y).cuda(),
            "y": y,
            "prob": y,
            "prob_1": y,
            "confidence": y,
            **metadata,
        }
    
    def eval_epoch(self, model, val_loader, test_loader, epoch):
        return
    
    def eval_step(self, model, batch):
        x, _, y, metadata = batch
        x = x.to(self.args.device)
        h = model.backbone(x)
        
        finetuner = LogisticRegression()
        finetuner.fit(h.cpu(), y.cpu())
        y_hat = finetuner.predict(h.cpu())
        loss = F.cross_entropy(y_hat, y)
        prob = F.softmax(y_hat, dim=1)
        
        return {
            "loss": F.cross_entropy(y_hat, y, reduction="none"),
            "y": y,
            "prob": prob,
            "confidence": prob.max(dim=1).values
            **metadata,
        }

    def create_model(self):
        from src.modeling.registry import resnet10

        backbone = ViT(image_size=256, patch_size=16,
            num_classes=2, dim=768, depth=6, heads=16,
            channels=1, mlp_dim=384)
        
        backbone.classifier.fc = nn.Identity()

        model = VICReg(
            backbone=backbone,
            proj_dims=128,
            features_dim=512,
            var_loss_weight=25.0,
            cov_loss_weight=1.0,
            inv_loss_weight=25.0,
        )

        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            return model.cuda()
        
        return model

    def get_features(self, model, x):
        return model.backbone(x)
    
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
        # core_splits = [
        #     remove_benign_cores_from_positive_patients(cores) for cores in core_splits
        # ]
        core_splits = [
            remove_cores_below_threshold_involvement(core, threshold_pct=40)
            for core in core_splits
        ]
        train_cores, val_cores, test_cores = core_splits
        train_cores = remove_benign_cores_from_positive_patients(train_cores)
        if self.args.benign_undersampling:
            train_cores = undersample_benign(train_cores)
        

        from src.data.exact.dataset.rf_datasets import SSLPatchesDatasetNew, PatchViewConfig
        
        from src.data.exact.transforms import TransformV3
        from src.data.exact.transforms import TensorImageAugmentation
        from src.data.exact.transforms import UltrasoundArrayAugmentation

        patch_view_cfg = PatchViewConfig(
            needle_region_only=True,
            prostate_region_only=False,
        )

        eval_transform = TransformV3()
        if args.augmentations_mode == "tensor_augs":
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
        
        print("Train cores:", train_cores)
        #print("Val cores:", val_cores)
        print("Test cores:", test_cores)

        train_dataset = SSLPatchesDatasetNew(
            patch_view_config=patch_view_cfg,
            core_specifier_list=train_cores,
            transform=train_transform,
            target_transform=self._label_transform,
        )
        val_dataset = SSLPatchesDatasetNew(
            patch_view_config=patch_view_cfg,
            core_specifier_list=val_cores,
            transform=eval_transform,
            target_transform=self._label_transform,
        )
        test_dataset = SSLPatchesDatasetNew(
            patch_view_config=patch_view_cfg,
            core_specifier_list=test_cores,
            transform=eval_transform,
            target_transform=self._label_transform,
        )

        return train_dataset, val_dataset, test_dataset
    
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
        if self.args.debug:
            wandb.init = lambda **kwargs: None
            wandb.log = lambda x: print(x)

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
            batch_size=self.args.batch_size if not self.args.eval_batch_size else self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
        self.test_loader = DataLoader(
            test_ds,
            batch_size=self.args.batch_size if not self.args.eval_batch_size else self.args.eval_batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

        torch.random.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

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

            #eval_metrics, eval_dfs = self.eval_epoch(
            #    model, self.val_loader, self.test_loader, epoch
            #)
            #metrics.update(eval_metrics)

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
            #for name, df in eval_dfs.items():
            #    df.to_csv(
            #        os.path.join(self.args.exp_dir, f"epoch_{epoch}", f"{name}.csv")
            #    )

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


class VICRegResNet10(VICRegExperiment):
    def create_model(self):
        from src.modeling.registry import resnet10

        backbone = resnet10()
        backbone.fc = nn.Identity()

        model = VICReg(
            backbone=backbone,
            proj_dims=128,
            features_dim=512,
            var_loss_weight=25.0,
            cov_loss_weight=1.0,
            inv_loss_weight=25.0,
        )

        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            return model.cuda()
        
        return model

class VICRegPVT(VICRegExperiment):
    def create_model(self):
        from projects.TRUS_ViT.models.pyramid_vit import PyramidVisionTransformerV2

        backbone = PyramidVisionTransformerV2()
        backbone.head = nn.Identity()

        model = VICReg(
            backbone=backbone,
            proj_dims=128,
            features_dim=512,
            var_loss_weight=25.0,
            cov_loss_weight=1.0,
            inv_loss_weight=25.0,
        )

        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            return model.cuda()
        
        return model

class VICRegCCT(VICRegExperiment):
    def create_model(self):
        #backbone = CCT(
        #    img_size = 256,
        #    embedding_dim = 384,
        #    n_conv_layers = 2,
        #    n_input_channels=1,
        #    kernel_size = 7,
        #    stride = 2,
        #    padding = 3,
        #    pooling_kernel_size = 3,
        #    pooling_stride = 2,
        #    pooling_padding = 1,
        #    num_layers = 8,
        #    num_heads = 6,
        #    mlp_ratio = 3,
        #    num_classes = 2,
        #    positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        #)

        # >>> r_params
        # 4900546
        # >>> c2_params
        # 1225603
        # >>> r_params/c2_params
        # 3.9984774841445394
        backbone = CCT(
            img_size = 256,
            embedding_dim = 128,
            n_conv_layers = 2,
            n_input_channels=1,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            pooling_kernel_size = 3,
            pooling_stride = 2,
            pooling_padding = 1,
            num_layers = 7,
            num_heads = 8,
            mlp_ratio = 3,
            num_classes = 2,
            positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        )
        
        backbone.classifier.fc = nn.Identity()

        model = VICReg(
            backbone=backbone,
            proj_dims=128,
            features_dim=128,
            var_loss_weight=25.0,
            cov_loss_weight=1.0,
            inv_loss_weight=25.0,
        )

        if torch.cuda.is_available():
            return model.cuda()
        
        return model
    
class VICRegViT(VICRegExperiment):
    def create_model(self):
        backbone = ViT(image_size=256, patch_size=16,
            num_classes=2, dim=768, depth=6, heads=16,
            channels=1, mlp_dim=2048)
        
        backbone.mlp_head[-1] = nn.Identity()

        model = VICReg(
            backbone=backbone,
            proj_dims=256,
            features_dim=768,
            var_loss_weight=25.0,
            cov_loss_weight=1.0,
            inv_loss_weight=25.0,
        )

        if torch.cuda.is_available():
            return model.cuda()
        
        return model

class VICReg(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        proj_dims: list,
        features_dim: int,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 1.0,
        inv_loss_weight: float = 25.0,
    ):
        super().__init__()
        self.backbone = backbone
        #self.projector = MLP(*[features_dim, *proj_dims])
        self.projector = nn.Linear(features_dim, proj_dims, bias=False)
        self.head = nn.Linear(features_dim, 2, bias=False)
        self.features_dim = features_dim
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight
        self.inv_loss_weight = inv_loss_weight

    def forward(self, X1, X2):
        X1 = self.backbone(X1)
        X2 = self.backbone(X2)

        X1 = self.projector(X1)
        X2 = self.projector(X2)

        loss = vicreg_loss_func(
            X1,
            X2,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
            sim_loss_weight=self.inv_loss_weight,
            #return_dict=True,
        )
        return loss