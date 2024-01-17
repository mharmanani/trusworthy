import hydra
from rich import print as rprint
import sys
import os


@hydra.main(config_path="config", config_name="config")
def main(args):
    rprint(args)

    if args.mode == "baseline":
        from projects.TRUS_ViT.base import BaseExperiment

        experiment = BaseExperiment(args)
        experiment.run()

    if args.mode == "effnet_b0":
        from projects.TRUS_ViT.models.efficient_net import EffNetB0Experiment

        experiment = EffNetB0Experiment(args)
        experiment.run()

    if args.mode == "resnet10":
        from projects.TRUS_ViT.models.resnet import ResNet10Experiment

        experiment = ResNet10Experiment(args)
        experiment.run()

    elif args.mode == "resnet18":
        from projects.TRUS_ViT.models.resnet import ResNet18Experiment

        experiment = ResNet18Experiment(args)
        experiment.run()

    elif args.mode == "vitus":
        from projects.TRUS_ViT.models.vit import ViTus

        experiment = ViTus(args)
        experiment.run()

    elif args.mode == "cctus":
        from projects.TRUS_ViT.models.vit import CCTus

        experiment = CCTus(args)
        experiment.run()

    elif args.mode == "trusformer":
        from projects.TRUS_ViT.models.trusformer import TRUSformerExperiment

        experiment = TRUSformerExperiment(args)
        experiment.run()

    elif args.mode == "multi_trusformer":
        from projects.TRUS_ViT.models.trusformer import MultiObjTRUSformerExperiment

        experiment = MultiObjTRUSformerExperiment(args)
        experiment.run()

    elif args.mode == "mask_trusformer":
        from projects.TRUS_ViT.models.mask_trusformer import MultiObjTRUSformerExperiment

        experiment = MultiObjTRUSformerExperiment(args)
        experiment.run()

    elif args.mode == "attn_mil":
        from projects.TRUS_ViT.models.attention_mil import AttentionMILExperiment

        experiment = AttentionMILExperiment(args)
        experiment.run()

    elif args.mode == "gattn_mil":
        from projects.TRUS_ViT.models.attention_mil import GatedAttentionMILExperiment

        experiment = GatedAttentionMILExperiment(args)
        experiment.run()

    elif args.mode == "levit":
        from projects.TRUS_ViT.models.vit import LeViTus

        experiment = LeViTus(args)
        experiment.run()

    elif args.mode == "pvtus":
        from projects.TRUS_ViT.models.vit import PVTus

        experiment = PVTus(args)
        experiment.run()
    
    elif args.mode == "ssl_cct":
        from projects.TRUS_ViT.models.vicreg import VICRegCCTus

        experiment = VICRegCCTus(args)
        experiment.run()

    elif args.mode == "ssl_vit":
        from projects.TRUS_ViT.models.vicreg import VICRegViTus

        experiment = VICRegViTus(args)
        experiment.run()

    elif args.mode == "ssl_pvt":
        from projects.TRUS_ViT.models.vicreg import VICRegPVT

        experiment = VICRegPVT(args)
        experiment.run()

    elif args.mode == "ssl_resnet10":
        from projects.TRUS_ViT.models.vicreg import VICRegResNet10

        experiment = VICRegResNet10(args)
        experiment.run()

    elif args.mode == "ssl_resnet50":
        from projects.TRUS_ViT.models.vicreg import VICRegResNet50

        experiment = VICRegResNet50(args)
        experiment.run()

if __name__ == "__main__":
    main()