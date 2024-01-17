import torch
from functools import partial

_DATASETS = {}


def register_dataset(f):
    _DATASETS[f.__name__] = f
    return f


def create_dataset(name, split="train", **kwargs):
    return _DATASETS[name](split, **kwargs)


def list_datasets():
    return list(_DATASETS.keys())


class Augmentations:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x).float(), self.transform(x).float()


@register_dataset
def exact_patches_ssl_all_centers_all_cores_all_patches_v0(split="train"):
    """
    Created by @pfrwilson
    ON 2023-02-04.

    Self-supervised dataset with all patches from all cores.
    normalized, and using crop-like augmentation. (SimCLR style)
    """

    from src.data.exact.splits import Splits
    from src.data import data_dir
    from torchvision import transforms as T
    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )

    transform = Augmentations(transform)

    splits = Splits(
        cohort_specifier="all",
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=False,
    )
    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=False
    )

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    return PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
    )


@register_dataset
def exact_patches_sl_all_centers_balanced_ndl(split="train"):
    from src.data.exact.splits import Splits, InvolvementThresholdFilter
    from src.data import data_dir
    from torchvision import transforms as T
    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    transform_ = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
        ]
    )
    transform = lambda x: transform_(x).float()

    splits = Splits(
        cohort_specifier=["UVA", "CRCEO", "PCC", "PMCC", "JH"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        merge_test_centers=True,
        merge_val_centers=True,
        undersample_benign_eval=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )
    from functools import partial

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    return PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
    )


@register_dataset
def exact_patches_uva_ndl(split="train"):
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
    )
    from src.data.exact.transforms import TransformV3

    t = TransformV3()
    splits = Splits(
        cohort_specifier=["UVA"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        undersample_benign_eval=True,
        merge_test_centers=True,
        merge_val_centers=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))
    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )
    from functools import partial

    cores = getattr(splits, f"get_{split}")()

    dataset = PatchesDatasetNew(
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=t,
        label_transform=partial(torch.tensor, dtype=torch.long),
    )
    return dataset


@register_dataset
def exact_patches_sl_tuffc_ndl(split="train"):
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
    )

    splits = Splits(
        cohort_specifier=["UVA600", "CRCEO428"],
        train_val_split_seed=2,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        merge_test_centers=True,
        merge_val_centers=True,
        undersample_benign_eval=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))
    splits.apply_filters(HasProstateMaskFilter())

    from src.data.exact.transforms import TransformV3

    transform = TransformV3()

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=True, needle_region_only=True
    )

    from src.data import data_dir
    from functools import partial

    cores = (
        splits.get_train()
        if split == "train"
        else splits.get_val()
        if split == "val"
        else splits.get_test()
    )

    out = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
    )
    out.num_classes = 2
    return out


@register_dataset
def exact_patches_ssl_tuffc_ndl(split="train"):
    dataset = exact_patches_sl_tuffc_ndl(split)

    from src.data.exact.splits import Splits
    from src.data import data_dir
    from torchvision import transforms as T
    from src.data.exact.transforms import MultiTransform
    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    augs = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )

    dataset.patch_transform = Augmentations(augs)
    return dataset


@register_dataset
def exact_patches_ssl_tuffc_prostate(split="train"):
    dataset = exact_patches_sl_tuffc_prostate(split)

    from src.data.exact.dataset import RF_PATCHES_MEAN, RF_PATCHES_STD

    augs = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(RF_PATCHES_MEAN, RF_PATCHES_STD),
            T.Resize((224, 224)),
            T.RandomResizedCrop((224, 224), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )

    dataset.patch_transform = Augmentations(augs)
    return dataset


@register_dataset
def exact_patches_sl_tuffc_prostate(split="train"):
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
    )

    splits = Splits(
        cohort_specifier=["UVA600", "CRCEO428"],
        train_val_split_seed=2,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        merge_test_centers=True,
        merge_val_centers=True,
        undersample_benign_eval=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))
    splits.apply_filters(HasProstateMaskFilter())

    from src.data.exact.transforms import TransformV3

    transform = TransformV3()

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=True, needle_region_only=False
    )

    from src.data import data_dir
    from functools import partial

    cores = (
        splits.get_train()
        if split == "train"
        else splits.get_val()
        if split == "val"
        else splits.get_test()
    )

    out = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
    )
    out.num_classes = 2
    return out


@register_dataset
def exact_patches_sl_tuffc_whole_image(split="train"):
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
    )

    splits = Splits(
        cohort_specifier=["UVA600", "CRCEO428"],
        train_val_split_seed=2,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        merge_test_centers=True,
        merge_val_centers=True,
        undersample_benign_eval=True,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))
    splits.apply_filters(HasProstateMaskFilter())

    from src.data.exact.transforms import TransformV3

    transform = TransformV3()

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=False
    )

    from src.data import data_dir
    from functools import partial

    cores = (
        splits.get_train()
        if split == "train"
        else splits.get_val()
        if split == "val"
        else splits.get_test()
    )

    out = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
    )
    out.num_classes = 2
    return out



@register_dataset
def exact_patches_sl_all_centers_balanced_ndl_instanceNorm(split='train'):
    from src.data import data_dir
    from torchvision import transforms as T    
    from src.data.exact.transforms import TransformV3
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
        )

     
    transform = TransformV3()

    merge_val_centers = True
    splits = Splits(
        cohort_specifier=["UVA", "CRCEO", "PCC", "PMCC", "JH"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        undersample_benign_eval=True,
        merge_test_centers=merge_val_centers,
        merge_val_centers=merge_val_centers,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )
    from functools import partial

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    
    if isinstance(cores, dict):
        dataset = {center: PatchesDatasetNew(
                                root=data_dir(),
                                core_specifier_list=cores_,
                                patch_view_config=patch_view_config,
                                patch_transform=transform,
                                label_transform=partial(torch.tensor, dtype=torch.long),
                                ) 
                   for center, cores_ in cores.items()
                   }
        
    else:
        dataset = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
        ) 
    
    
    return dataset 


@register_dataset
def exact_patches_sl_uva_centers_balanced_ndl_instanceNorm(split='train'):
    from src.data import data_dir
    from torchvision import transforms as T    
    from src.data.exact.transforms import TransformV3
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
        )

     
    transform = TransformV3()

    merge_val_centers = True
    splits = Splits(
        cohort_specifier=["UVA"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        undersample_benign_eval=True,
        merge_test_centers=merge_val_centers,
        merge_val_centers=merge_val_centers,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=False, needle_region_only=True
    )
    from functools import partial

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    
    if isinstance(cores, dict):
        dataset = {center: PatchesDatasetNew(
                                root=data_dir(),
                                core_specifier_list=cores_,
                                patch_view_config=patch_view_config,
                                patch_transform=transform,
                                label_transform=partial(torch.tensor, dtype=torch.long),
                                ) 
                   for center, cores_ in cores.items()
                   }
        
    else:
        dataset = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
        ) 
    
    
    return dataset 


@register_dataset
def exact_patches_sl_uva600_centers_balanced_ndl_prst_instanceNorm(split='train'):
    from src.data import data_dir
    from torchvision import transforms as T    
    from src.data.exact.transforms import TransformV3
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
        )

     
    transform = TransformV3()

    merge_val_centers = True
    splits = Splits(
        cohort_specifier=["UVA600"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        undersample_benign_eval=True,
        merge_test_centers=merge_val_centers,
        merge_val_centers=merge_val_centers,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=True, needle_region_only=True
    )
    from functools import partial

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    
    if isinstance(cores, dict):
        dataset = {center: PatchesDatasetNew(
                                root=data_dir(),
                                core_specifier_list=cores_,
                                patch_view_config=patch_view_config,
                                patch_transform=transform,
                                label_transform=partial(torch.tensor, dtype=torch.long),
                                ) 
                   for center, cores_ in cores.items()
                   }
        
    else:
        dataset = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
        ) 
    
    
    return dataset 


@register_dataset
def exact_patches_ssl_uva600_centers_balanced_prst_instanceNorm(split='train'):
    from src.data import data_dir
    from torchvision import transforms as T    
    from src.data.exact.transforms import TransformV3
    from src.data.exact.splits import (
        Splits,
        InvolvementThresholdFilter,
        HasProstateMaskFilter,
        )


    augment_transform = T.Compose(
        [
            T.RandomResizedCrop((256, 256), scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomInvert(),
        ]
    )
    augment_transform = Augmentations(augment_transform)
    transform = TransformV3(tensor_transform=augment_transform)

    merge_val_centers = True
    splits = Splits(
        cohort_specifier=["UVA600"],
        train_val_split_seed=0,
        train_val_ratio=0.2,
        undersample_benign_train=True,
        undersample_benign_eval=True,
        merge_test_centers=merge_val_centers,
        merge_val_centers=merge_val_centers,
    )
    splits.apply_filters(InvolvementThresholdFilter(0.4))

    from src.data.exact.dataset.rf_datasets import PatchesDatasetNew
    from src.data.exact.core import PatchViewConfig

    patch_view_config = PatchViewConfig(
        prostate_region_only=True, needle_region_only=False, patch_strides=(3, 3)
    )
    from functools import partial

    cores = None
    if split == "train":
        cores = splits.get_train()
    elif split == "val":
        cores = splits.get_val()
    elif split == "test":
        cores = splits.get_test()
    else:
        raise ValueError(f"Unknown split {split}")
    
    if isinstance(cores, dict):
        dataset = {center: PatchesDatasetNew(
                                root=data_dir(),
                                core_specifier_list=cores_,
                                patch_view_config=patch_view_config,
                                patch_transform=transform,
                                label_transform=partial(torch.tensor, dtype=torch.long),
                                ) 
                   for center, cores_ in cores.items()
                   }
        
    else:
        dataset = PatchesDatasetNew(
        root=data_dir(),
        core_specifier_list=cores,
        patch_view_config=patch_view_config,
        patch_transform=transform,
        label_transform=partial(torch.tensor, dtype=torch.long),
        ) 
    
    
    return dataset 


@register_dataset
def cifar10(split="train"):
    from torchvision.datasets import CIFAR10
    from torchvision import transforms as T
    from src.data import data_dir

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    return CIFAR10(
        root=data_dir(), download=True, train=split == "train", transform=transform
    )


from torchvision import transforms as T


class CIFAR10Augmentations:
    def __init__(self):
        self.augmentations = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )

    def __call__(self, x):
        return self.augmentations(x), self.augmentations(x)


@register_dataset
def cifar10_ssl_mode(split="train"):
    from torchvision.datasets import CIFAR10
    from src.data import data_dir

    transform = CIFAR10Augmentations()

    return CIFAR10(
        root=data_dir(), download=True, train=split == "train", transform=transform
    )


@register_dataset
def cifar10_fast(split="train"):
    from src.data.benchmarks.cifar10 import FastCIFAR10
    from torchvision import transforms as T
    from src.data import data_dir

    transform = T.Compose(
        [
            T.Lambda(lambda x: x.transpose(2, 1, 0)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    return FastCIFAR10(
        root="data",
        train=split == "train",
        transform=transform,
        target_transform=partial(torch.tensor, dtype=torch.long),
    )


@register_dataset
def ijcars_2023_dataset(split="train"):
    ...
