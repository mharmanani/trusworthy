# TRUSWorthy: Transformer Ensembles for Confident Detection of Prostate Cancer in Micro-Ultrasound

## Preliminary
This code represents a significant portion of the code used to generate the results for our paper. The models are implemented in `projects/TRUS_ViT/models`. There are several types of models implemented:

- The ResNet (dubbed `ResNet10` in the code) model is implemented in `src/modeling/registry.py`, and a wrapper `ResNet10Experiment` is provided in `projects/TRUS_ViT/models/resnet.py` for training, validation, and testing.

- SSL+ResNet uses the same ResNet10 backbone as above, but is trained using the logic of `projects/TRUS_ViT/models/vicreg.py`. 

- The TRUSformer architecture uses the SSL+ResNet architecture as a feature extractor, and the `TransformerEncoder` (BERT) architecture implemented in `src/modeling/bert` as a MIL aggregator. 

## Training
To train a single-model architecture, we use the bash scripts located in `projects/TRUS_ViT/job_submission`. These scripts parse a set of hyperparameters outlined in a `.yaml` file in `projects/TRUS_ViT/experiments`. 

For example, to train a ResNet