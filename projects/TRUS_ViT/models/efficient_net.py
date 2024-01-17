from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment

from torch import nn

class EffNetB0Experiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(self, pretrained=False):
        from torchvision.models.efficientnet import efficientnet_b0

        model = efficientnet_b0()
        model.features[0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if pretrained:
            model.classifier = nn.Identity()
            ssl_weights = torch.load("/h/harmanan/checkpoint/vicreg_weights/needle/resnet10/fold{0}.pth".format(self.args.fold))
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            model.load_state_dict(bb_weights)
            model.classifier = nn.Linear(1408, 2)

        return model.cuda()
    
class EffNetB2Experiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(self, pretrained=False):
        from torchvision.models.efficientnet import efficientnet_b2

        model = efficientnet_b2()

        if pretrained:
            model.fc = nn.Identity()
            ssl_weights = torch.load("/h/harmanan/checkpoint/vicreg_weights/resnet18/fold0.pth")
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            model.load_state_dict(bb_weights)
            model.fc = nn.Linear(512, 2)

        return resnet18().cuda()