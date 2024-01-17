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

    if args.mode == "resnet10":
        from projects.TRUS_ViT.models.resnet import ResNet10Experiment

        experiment = ResNet10Experiment(args)
        experiment.run()

    elif args.mode == "trusformer":
        from projects.TRUS_ViT.models.trusformer import TRUSformerExperiment

        experiment = TRUSformerExperiment(args)
        experiment.run()

    elif args.mode == "ssl_resnet10":
        from projects.TRUS_ViT.models.vicreg import VICRegResNet10

        experiment = VICRegResNet10(args)
        experiment.run()

if __name__ == "__main__":
    main()