import sys
sys.path.append('..')
sys.path.append('../..')

#from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment
from projects.TRUS_ViT.models.pyramid_vit import PyramidVisionTransformerV2
from src.modeling.registry import resnet10, resnet18
from src.modeling.bert import TransformerEncoder
from src.modeling.attention import AttentionMIL, GatedAttentionMIL
from src.modeling.positional_embedding import GridPositionEmbedder2d
from vit_pytorch.vit import ViT
from vit_pytorch.cct import CCT

import torch
from torch import nn
import torch.nn.functional as F

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

class MILModel(nn.Module):
    def unpack_bag(self, bag):
        xs = []
        for i in range(self.bag_dim):
            x = bag[:, i, :, :].unsqueeze(1)
            xs.append(x)

        return tuple(xs)

    def forward(self, bag, pos):
        xs = self.unpack_bag(bag)
        hs = []
        for i in range(len(xs)):
            h = self.model(xs[i].cuda())
            h = self.projector(h)
            hs.append(h)

        pos_mx = torch.stack([torch.Tensor([pos[i][0][0].item(), pos[i][1][0].item()]) for i in range(self.bag_dim)], dim=0)
        pos_emb = self.pos_embedding(pos_mx[:, 0].long(), pos_mx[:, 1].long())

        h = torch.stack(hs, dim=1)
        h = h + pos_emb
        h = self.bert(h).last_hidden_state
        h = h.reshape(h.shape[0], -1)
        y = self.classifier(h)

        return y

class MultiObjMILModel(MILModel):
    def forward(self, bag, pos):
        xs = self.unpack_bag(bag)
        hs = []
        zs = []
        for i in range(len(xs)):
            h = self.model(xs[i].cuda())
            z = self.roi_clf(h)
            h = self.projector(h)
            hs.append(h)
            zs.append(z)

        pos_mx = torch.stack([torch.Tensor([pos[i][0][0].item(), pos[i][1][0].item()]) for i in range(self.bag_dim)], dim=0)
        pos_emb = self.pos_embedding(pos_mx[:, 0].long(), pos_mx[:, 1].long())

        h = torch.stack(hs, dim=1)
        z = torch.stack(zs, dim=1)

        h = h + pos_emb
        h = self.bert(h).last_hidden_state
        h = h.reshape(h.shape[0], -1)
        y = self.classifier(h)

        return y, z

class AttentionMILModel(MILModel):
    def __init__(self, fold=0, bag_dim=55, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = resnet10()
        #self.model.fc = nn.Identity()

        if pretrained:
            patch_weights = torch.load('/h/harmanan/checkpoint/mus_patch_resnet_needle/fold{0}.pth'.format(fold))
            patch_weights = torch.load('/h/harmanan/checkpoint/tmi23/resnet.pth')
            self.model.load_state_dict(patch_weights)

        # Freeze the model
        # for param in self.model.parameters():
        #    param.requires_grad = False

        self.model.fc = nn.Identity()
        
        self.emb_dim = 512
        self.proj_dim = 512

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.classifier = AttentionMIL(
            input_dim=self.proj_dim,
            num_classes=2
        )
        
    def forward(self, bag, pos):
        xs = self.unpack_bag(bag)
        hs = []
        for i in range(len(xs)):
            h = self.model(xs[i].cuda())
            h = self.projector(h)
            hs.append(h)

        h = torch.stack(hs, dim=1)
        y = self.classifier(h)

        return y

class GatedAttentionMILModel(MILModel):
    def __init__(self, fold=0, bag_dim=55, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = resnet10()
        #self.model.fc = nn.Identity()

        if pretrained:
            # patch_weights = torch.load('/h/harmanan/checkpoint/mus_patch_resnet_needle/fold{0}.pth'.format(fold))
            patch_weights = torch.load('/h/harmanan/checkpoint/tmi23/fold{0}.pth'.format(fold))
            patch_weights = torch.load('/h/harmanan/checkpoint/tmi23/resnet.pth')
            self.model.load_state_dict(patch_weights)

        # Freeze the model
        # for param in self.model.parameters():
        #    param.requires_grad = False

        self.model.fc = nn.Identity()
        
        self.emb_dim = 512
        self.proj_dim = 512

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.classifier = GatedAttentionMIL(
            input_dim=self.proj_dim,
            num_classes=2
        )
        
    def forward(self, bag, pos):
        xs = self.unpack_bag(bag)
        hs = []
        for i in range(len(xs)):
            h = self.model(xs[i].cuda())
            h = self.projector(h)
            hs.append(h)

        h = torch.stack(hs, dim=1)
        y = self.classifier(h)

        return y

       
class AttentionMILExperiment(BaseExperiment):
    def create_model(self):
        model = AttentionMILModel(fold=self.args.fold, 
                                  bag_dim=self.args.num_patches,
                                  pretrained=self.args.pretrained)
        
        return model.cuda()
    
    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        bag, y, pos, metadata = batch
        y = ((y.float() / 100) != 0).long().to(self.args.device)
        y_hat = model(bag, pos).to(self.args.device)
        prob = F.softmax(y_hat, dim=1)
        loss = F.cross_entropy(y_hat, y.long())
        loss.backward()
        optimizer.step()

        return {
            "loss": F.cross_entropy(y_hat, y.long(), reduction="none"),
            "y": y,
            "prob": prob,
            "pred": y_hat,
            **metadata,
        }
    
    def eval_step(self, model, batch):
        bag, y, pos, metadata = batch
        y = ((y.float() / 100) != 0).long().to(self.args.device)
        y_hat = model(bag, pos).to(self.args.device)
        prob = F.softmax(y_hat, dim=1)

        return {
            "loss": F.cross_entropy(y_hat, y.long(), reduction="none"),
            "y": y,
            "prob": prob,
            "pred": y_hat,
            **metadata,
        }

    def create_datasets(self, args):
        from src.data.exact.cohort_selection import (
            get_cores_for_patients,
            get_patient_splits,
            get_tmi23_patient_splits,
            remove_benign_cores_from_positive_patients,
            remove_cores_below_threshold_involvement,
            undersample_benign,
            undersample_benign_as_kfold,
        )

        if self.args.cross_val:
            patient_splits = get_patient_splits(fold=self.args.fold)
        else: patient_splits = get_tmi23_patient_splits()
        
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
            train_cores = undersample_benign(train_cores)

        from src.data.exact.dataset.rf_datasets import BagOfPatchesDatasetNew, PatchViewConfig
        
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

        from torch.utils.data import Subset

        train_dataset = BagOfPatchesDatasetNew(
            core_specifier_list=train_cores,
            patch_view_config=patch_view_cfg,
            transform=train_transform,
            target_transform=self._label_transform,
            patch_increments=round(55/self.args.num_patches),
        )
        val_dataset = BagOfPatchesDatasetNew(
            core_specifier_list=val_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=self._label_transform,
            patch_increments=round(55/self.args.num_patches),
        )
        test_dataset = BagOfPatchesDatasetNew(
            core_specifier_list=test_cores,
            patch_view_config=patch_view_cfg,
            transform=eval_transform,
            target_transform=self._label_transform,
            patch_increments=round(55/self.args.num_patches),
        )

        if self.args.debug:
            indices = train_dataset.benign_indices[:5] + train_dataset.cancer_indices[:5]
            train_dataset = Subset(train_dataset, indices)

            indices = val_dataset.benign_indices[:5] + val_dataset.cancer_indices[:5]
            val_dataset = Subset(val_dataset, indices)

            indices = test_dataset.benign_indices[:5] + test_dataset.cancer_indices[:5]
            test_dataset = Subset(test_dataset, indices)
        
        return train_dataset, val_dataset, test_dataset
    

class GatedAttentionMILExperiment(AttentionMILExperiment):
    def create_model(self):
        model = GatedAttentionMILModel(fold=self.args.fold, 
                                       bag_dim=self.args.num_patches,
                                       pretrained=self.args.pretrained)
        
        return model.cuda()