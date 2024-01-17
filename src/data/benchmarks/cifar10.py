from torchvision.datasets import CIFAR10
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import os
from torchvision import transforms as T


def cifar10_augmentations():
    return T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


class CIFAR10DataModule(LightningDataModule):
    def __init__(self, train_transform, val_transform, batch_size=64, num_workers=8):
        super().__init__()
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=self.train_transform,
            target_transform=self._target_transform,
        )
        self.val_dataset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=self.val_transform,
            target_transform=self._target_transform,
        )

    def _target_transform(self, target):
        return torch.tensor(target).long()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class FastCIFAR10(Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None):
        root = Path(root)
        if not (root / "cifar-10-fast").is_dir():
            os.makedirs(root / "cifar-10-fast")
            self._make_dataset(root)
        if train:
            self.data = np.load(
                root / "cifar-10-fast" / "train_data.npy", mmap_mode="r"
            )
            self.labels = np.load(root / "cifar-10-fast" / "train_labels.npy")
        else:
            self.data = np.load(root / "cifar-10-fast" / "test_data.npy", mmap_mode="r")
            self.labels = np.load(root / "cifar-10-fast" / "test_labels.npy")
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def _make_dataset(self, root):
        print("making datasets")
        train_set = CIFAR10(root, train=True, download=True)
        train_data = np.zeros((len(train_set), 3, 32, 32), dtype=np.uint8)
        train_labels = np.zeros(len(train_set), dtype=np.int64)
        for i, (img, _) in enumerate(train_set):
            img = np.array(img).astype(np.uint8).transpose((2, 0, 1))
            train_data[i] = img
            train_labels[i] = train_set[i][1]
        np.save(root / "cifar-10-fast" / "train_data.npy", train_data)
        np.save(root / "cifar-10-fast" / "train_labels.npy", train_labels)

        test_set = CIFAR10(root, train=False, download=True)
        test_data = np.zeros((len(test_set), 3, 32, 32), dtype=np.uint8)
        test_labels = np.zeros(len(test_set), dtype=np.int64)
        for i, (img, _) in enumerate(test_set):
            img = np.array(img).astype(np.uint8).transpose((2, 0, 1))
            test_data[i] = img
            test_labels[i] = test_set[i][1]
        np.save(root / "cifar-10-fast" / "test_data.npy", test_data)
        np.save(root / "cifar-10-fast" / "test_labels.npy", test_labels)
