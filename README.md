# TRUSWorthy: Transformer Ensembles for Confident Detection of Prostate Cancer in Micro-Ultrasound

![TRUSWorthy](figs/trusworthy.png)

## Preliminary
This code represents a significant portion of the code used to generate the results for our paper. The models are implemented in `projects/TRUS_ViT/models`. There are several types of models implemented:

- The ResNet (dubbed `ResNet10` in the code) model is implemented in `src/modeling/registry.py`, and a wrapper `ResNet10Experiment` is provided in `projects/TRUS_ViT/models/resnet.py` for training, validation, and testing.

- SSL+ResNet uses the same ResNet10 backbone as above, but is trained using the logic of `projects/TRUS_ViT/models/vicreg.py`. 

- The TRUSformer architecture uses the SSL+ResNet architecture as a feature extractor, and the `TransformerEncoder` (BERT) architecture implemented in `src/modeling/bert.py` as a MIL aggregator. 

## Training a model
### Single-model baselines
To train a single-model architecture, we use the bash scripts located in `projects/TRUS_ViT/job_submission`. These scripts parse a set of hyperparameters outlined in a `.yaml` file in `projects/TRUS_ViT/experiments`. For example:

```yaml
augmentations_mode: tensor_augs
batch_size: 64
device: cuda

fold: 0
lr: 3e-4
model_name: resnet10
num_epochs: 25

seed: 42
```

### Ensemble baselines

To train an $n$-Deep Ensemble, we need to initialize each model with a different seed. We can set the hyperparameters inside the job submission such as:

```bash
python main.py experiment=resnet10 seed=42 fold=0
python main.py experiment=resnet10 seed=81 fold=0
python main.py experiment=resnet10 seed=881 fold=0
python main.py experiment=resnet10 seed=392 fold=0
python main.py experiment=resnet10 seed=659 fold=0
```

The code above trains an ensemble of 5 ResNets on folds 1,2,3,4 and evaluates it on fold 0.

We can also train an ensemble of TRUSformers using:
```bash
python main.py experiment=trusformer seed=42 fold=0
python main.py experiment=trusformer seed=81 fold=0
python main.py experiment=trusformer seed=881 fold=0
python main.py experiment=trusformer seed=392 fold=0
python main.py experiment=trusformer seed=659 fold=0
```

### Mixed deep ensembles
We can train a TRUSWorthy model using the same code presented above. The only difference is that we are also resampling the set of benign cores seen by the model during training using the same seed specified in the experiment `.yaml` file. 

```bash
python main.py experiment=trusformer mix_ens=true seed=42 fold=0
python main.py experiment=trusformer mix_ens=true seed=81 fold=0
python main.py experiment=trusformer mix_ens=true seed=881 fold=0
python main.py experiment=trusformer mix_ens=true seed=392 fold=0
python main.py experiment=trusformer mix_ens=true seed=659 fold=0
```