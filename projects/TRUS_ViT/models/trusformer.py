import sys
sys.path.append('..')
sys.path.append('../..')

#from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment
from projects.TRUS_ViT.models.pyramid_vit import PyramidVisionTransformerV2
from src.modeling.registry import resnet10, resnet18
from src.modeling.bert import TransformerEncoder
from src.modeling.attention import AttentionMIL
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

backbones = {
    'resnet': resnet10(),
    'vit': ViT(image_size=256, patch_size=16,
            num_classes=2, dim=768, depth=6, heads=16,
            channels=1, mlp_dim=2048),
    'cct': CCT(
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
        ),
    'pvt': PyramidVisionTransformerV2()
}


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
        h = h.mean(dim=1)
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
        h = h.mean(dim=1)
        h = h.reshape(h.shape[0], -1)
        y = self.classifier(h)

        return y, z

class MultiObjTRUSformerResNet(MultiObjMILModel):
    def __init__(self, backbone='resnet', bag_dim=19, fold=-1, centerwise_split=False, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]

        if pretrained:
            if centerwise_split:
                patch_weights = torch.load('/h/harmanan/checkpoint/vicreg_centerwise/finetune/fold{0}.pth'.format(fold))
            else:
                patch_weights = torch.load('/h/harmanan/checkpoint/mus_patch_resnet_needle/fold{0}.pth'.format(fold))
            self.model.load_state_dict(patch_weights)
         
        self.roi_clf = nn.Linear(512, 2)
        self.model.fc = nn.Identity() 

        self.emb_dim = 512
        self.proj_dim = 512

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim,
                                        num_hidden_layers=12,
                                        num_attention_heads=8,
                                        intermediate_size=self.proj_dim)
        
        self.classifier = nn.Linear(self.proj_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

class TRUSformerResNet(MILModel):
    def __init__(self, backbone='resnet', bag_dim=19, fold=-1, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]
        #self.model.fc = nn.Identity()

        if pretrained:
            if fold >= 0:
                patch_weights = torch.load('/h/harmanan/checkpoint/mus_patch_resnet_needle/fold{0}.pth'.format(fold))
            else:
                patch_weights = torch.load('/h/harmanan/checkpoint/tmi23/resnet.pth')
            # patch_weights = torch.load('/checkpoint/harmanan/10679956/epoch_10.pth')
            self.model.load_state_dict(patch_weights)

        # Freeze the model
        # for param in self.model.parameters():
        #    param.requires_grad = False

        self.model.fc = nn.Identity()
        
        self.emb_dim = 512
        self.proj_dim = 512

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim, 
                                       num_hidden_layers=12, 
                                       num_attention_heads=8, 
                                       intermediate_size=self.proj_dim)

        self.classifier = nn.Linear(self.proj_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

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
        print(h.shape)
        h = self.bert(h).last_hidden_state
        h = h.mean(dim=1)
        h = h.reshape(h.shape[0], -1)
        y = self.classifier(h)

        return y

class TRUSformerViT(MILModel):
    def __init__(self, backbone='vit', bag_dim=19, fold=0, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]
        self.model.mlp_head[-1] = nn.Identity()

        if pretrained:
            patch_weights = torch.load('/h/harmanan/checkpoint/vicreg_weights/needle/vit/fold{0}.pth'.format(fold))
            parse_weights = {}
            for key, value in patch_weights.items():
                if 'backbone' in key:
                    parse_weights[key.replace('backbone.', '')] = value
            self.model.load_state_dict(parse_weights)
        
        
        self.emb_dim = 768
        self.proj_dim = 72

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim, 
                                       num_hidden_layers=12, 
                                       num_attention_heads=8, 
                                       intermediate_size=self.proj_dim)

        self.classifier = nn.Linear(self.proj_dim * self.bag_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

class TRUSformerPvT(MILModel):
    def __init__(self, backbone='pvt', bag_dim=19, fold=0, pretrained=True):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]

        if pretrained:
            self.model.head = nn.Identity()
            ssl_weights = torch.load('/h/harmanan/checkpoint/vicreg_weights/needle/pvt/fold{0}.pth'.format(fold))
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            self.model.load_state_dict(bb_weights)

        self.emb_dim = 512
        self.proj_dim = 72

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim, 
                                       num_hidden_layers=4, 
                                       num_attention_heads=8, 
                                       intermediate_size=self.proj_dim)

        self.classifier = nn.Linear(self.proj_dim * self.bag_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

class MultiObjTRUSformerCCT(MultiObjMILModel):
    def __init__(self, backbone='cct', bag_dim=6, fold=0, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]

        if pretrained:
            patch_weights = torch.load('/h/harmanan/checkpoint/resume_small_cct/fold{0}.pth'.format(fold))
            self.model.load_state_dict(patch_weights)

        # Freeze the model
        # for param in self.model.parameters():
        #   param.requires_grad = False

        self.roi_clf = nn.Linear(128, 2)
        self.model.classifier.fc = nn.Identity()

        self.emb_dim = 128
        self.proj_dim = 128

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim,
                                        num_hidden_layers=12,
                                        num_attention_heads=8,
                                        intermediate_size=self.proj_dim)
        
        self.classifier = nn.Linear(self.proj_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

class TRUSformerCCT(MILModel):
    def __init__(self, backbone='cct', bag_dim=6, fold=0, pretrained=False):
        super().__init__()
        self.bag_dim = bag_dim

        self.model = backbones[backbone]

        if pretrained:
            patch_weights = torch.load('/h/harmanan/checkpoint/resume_small_cct/fold{0}.pth'.format(fold))
            self.model.load_state_dict(patch_weights)
                
        self.model.classifier.fc = nn.Identity()

        # freeze the model
        # for param in self.model.parameters():
        #    param.requires_grad = False
        
        self.emb_dim = 128
        self.proj_dim = 128

        self.projector = nn.Linear(self.emb_dim, self.proj_dim)
        self.bert = TransformerEncoder(hidden_size=self.proj_dim, 
                                       num_hidden_layers=12, 
                                       num_attention_heads=8, 
                                       intermediate_size=self.proj_dim)

        self.classifier = nn.Linear(self.proj_dim, 2)

        grid_shape = (28, 46)
        self.pos_embedding = GridPositionEmbedder2d(self.proj_dim, grid_shape)

class TRUSformerExperiment(BaseExperiment):
    def create_model(self):
        if self.args.backbone == 'resnet':
            model = TRUSformerResNet(fold=self.args.fold, backbone=self.args.backbone, 
                                     bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        elif self.args.backbone == 'vit':
            model = TRUSformerViT(fold=self.args.fold, backbone=self.args.backbone, 
                                  bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        
        elif 'cct' in self.args.backbone:
            model = TRUSformerCCT(fold=self.args.fold, backbone=self.args.backbone, 
                                  bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        elif self.args.backbone == 'pvt':
            model = TRUSformerPvT(fold=self.args.fold, backbone=self.args.backbone, 
                                  bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        else:
            raise ValueError("Unknown backbone")
        
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
                train_cores = undersample_benign_as_kfold(train_cores, 
                                                          benign_to_cancer_ratio=self.args.benign_to_cancer_ratio)[self.args.undersample_fold_idx]
            else:
                train_cores = undersample_benign(train_cores, 
                                                 #seed=self.args.seed, 
                                                 benign_to_cancer_ratio=self.args.benign_to_cancer_ratio)

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
    

class MultiObjTRUSformerExperiment(TRUSformerExperiment):
    def create_model(self):
        if self.args.backbone == 'resnet':
            model = MultiObjTRUSformerResNet(fold=self.args.fold, backbone=self.args.backbone,
                                             bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        elif self.args.backbone == 'vit':
            model = TRUSformerViT(fold=self.args.fold, backbone=self.args.backbone, 
                                  bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        
        elif 'cct' in self.args.backbone:
            model = MultiObjTRUSformerCCT(fold=self.args.fold, backbone=self.args.backbone,
                                          bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        elif self.args.backbone == 'pvt':
            model = TRUSformerPvT(fold=self.args.fold, backbone=self.args.backbone, 
                                  bag_dim=self.args.num_patches, pretrained=self.args.pretrained)
        else:
            raise ValueError("Unknown backbone")
        
        return model.cuda()
    
    def train_step(self, model, batch, optimizer, epoch):
        optimizer.zero_grad()
        bag, y, pos, metadata = batch
        y = ((y.float() / 100) != 0).long().to(self.args.device)
        y_hat, z_hat = model(bag, pos)
        y_hat = y_hat.to(self.args.device)
        z_hat = z_hat.to(self.args.device)
        z = torch.stack([y] * self.args.num_patches, dim=1)

        z_hat = z_hat.reshape(-1, 2)
        z = z.reshape(-1)

        gamma = self.args.gamma

        prob = F.softmax(y_hat, dim=1)
        loss = gamma * F.cross_entropy(y_hat, y.long()) + (1-gamma) * F.cross_entropy(z_hat, z.long())
        loss.backward()
        optimizer.step()

        L2_non_redux = F.cross_entropy(z_hat, z.long(), reduction="none")
        L2_non_redux = L2_non_redux.reshape(-1, self.args.num_patches)
        L2_non_redux = torch.mean(L2_non_redux, dim=1)

        return {
            "loss": gamma * F.cross_entropy(y_hat, y.long(), reduction="none") + (1-gamma) * L2_non_redux,
            "y": y,
            "prob": prob,
            "pred": y_hat,
            **metadata,
        }
    
    def eval_step(self, model, batch):
        bag, y, pos, metadata = batch
        y = ((y.float() / 100) != 0).long().to(self.args.device)
        y_hat, z_hat = model(bag, pos)
        y_hat = y_hat.to(self.args.device)
        z_hat = z_hat.to(self.args.device)
        z = torch.stack([y] * self.args.num_patches, dim=1)

        z_hat = z_hat.reshape(-1, 2)
        z = z.reshape(-1)

        gamma = self.args.gamma

        prob = F.softmax(y_hat, dim=1)

        L2_non_redux = F.cross_entropy(z_hat, z.long(), reduction="none")
        L2_non_redux = L2_non_redux.reshape(-1, self.args.num_patches)
        L2_non_redux = torch.mean(L2_non_redux, dim=1)

        return {
            "loss": gamma * F.cross_entropy(y_hat, y.long(), reduction="none") + (1-gamma) * L2_non_redux,
            "y": y,
            "prob": prob,
            "pred": y_hat,
            **metadata,
        }