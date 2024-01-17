from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment

from src.modeling.registry import resnet10, resnet18

from vit_pytorch.vit import ViT
from vit_pytorch.levit import LeViT
from vit_pytorch.cct import CCT
from vit_pytorch.distill import DistillableViT, DistillWrapper
from vit_pytorch.recorder import Recorder
from vit_pytorch.extractor import Extractor

from projects.TRUS_ViT.models.pyramid_vit import PyramidVisionTransformerV2

from torch.nn import functional as F


class ViTExperiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)
    
    def create_model(self):
        model = ViT(image_size=256, patch_size=8,
            num_classes=2, dim=1024, depth=6, heads=16,
            channels=1, mlp_dim=2048, dropout=0.1)

        return model.cuda()
    
    def get_attention_map(self, model, x):
        v = Recorder(model)
        _, attns = v(x)
        attns = attns.cpu().detach().numpy()
        return attns
    
    def get_features(self, model, x):
        v = Extractor(model)
        features = v(x)
        _, features = features.cpu().detach().numpy()
        return features

class ViTus(ViTExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)
    
    def create_model(self, pretrained=True):
        model = ViT(image_size=256, patch_size=16,
            num_classes=2, dim=768, depth=6, heads=16,
            channels=1, mlp_dim=2048)
        
        if pretrained:
            model.mlp_head[-1] = nn.Identity()
            ssl_weights = torch.load("/h/harmanan/checkpoint/vicreg_weights/vit/fold0.pth")
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            model.load_state_dict(bb_weights)
            model.mlp_head[-1] = nn.Linear(768, 2)

        return model.cuda()
    
    def get_attention_map(self, model, x):
        v = Recorder(model)
        _, attns = v(x)
        attns = attns.cpu().detach().numpy()
        return attns
    
    def get_features(self, model, x):
        v = Extractor(model)
        features = v(x)
        _, features = features.cpu().detach().numpy()
        return features

class CCTus(ViTExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(self, pretrained=True):
        cct = CCT(
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

        if self.args.pretrained:
            # cct.classifier.fc = nn.Identity()
            # ssl_weights = torch.load('/h/harmanan/checkpoint/new_vicreg/cct/fold{0}.pth'.format(self.args.fold))
            # bb_weights = {}
            # print(cct.state_dict().keys())
            # for key, value in ssl_weights.items():
            #     if 'backbone' in key:
            #         bb_weights[key.replace('backbone.', '')] = value
            # cct.load_state_dict(bb_weights)
            # cct.classifier.fc = nn.Linear(128, 2)

            slurmid_per_fold = [10695224, 10695225, 10695226, 10695227, 10695228]
            weights = torch.load('/h/harmanan/checkpoint/new_models/cct/fold{0}.pth'.format(self.args.fold))
            weights = torch.load(f"/checkpoint/harmanan/{slurmid_per_fold[self.args.fold]}/epoch_15.pth")
            cct.load_state_dict(weights)

        return cct.cuda()
  
class PVTus(BaseExperiment):
    def create_model(self, pretrained=True):
        pvt = PyramidVisionTransformerV2()

        if pretrained:
            pvt.head = nn.Identity()
            ssl_weights = torch.load('/h/harmanan/checkpoint/vicreg_weights/needle/pvt/fold{0}.pth'.format(self.args.fold))
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            pvt.load_state_dict(bb_weights)
            pvt.head = nn.Linear(512, 2)

        return pvt.cuda()