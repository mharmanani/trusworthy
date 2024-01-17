from projects.TRUS_ViT.utils import *
from projects.TRUS_ViT.base import BaseExperiment

class ResNet10Experiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(self, pretrained=False):
        from src.modeling.registry import resnet10

        model = resnet10()

        if pretrained:
            model.fc = nn.Identity()
            if self.args.kfold_centerwise:
                ssl_weights = torch.load("/h/harmanan/checkpoint/vicreg_centerwise/pretrain/fold{0}.pth".format(self.args.fold))
            else:
                ssl_weights = torch.load("/h/harmanan/checkpoint/vicreg_weights/needle/resnet10/fold{0}.pth".format(self.args.fold))
            bb_weights = {}
            for key, value in ssl_weights.items():
                if 'backbone' in key:
                    bb_weights[key.replace('backbone.', '')] = value
            model.load_state_dict(bb_weights)
            model.fc = nn.Linear(512, 2)

        return model.cuda()
    
class ResNet18Experiment(BaseExperiment):
    def train_epoch(self, model, train_loader, optimizer, epoch):
        return super().train_epoch(model, train_loader, optimizer, epoch)

    def create_model(self, pretrained=False):
        from src.modeling.registry import resnet18

        model = resnet18()

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