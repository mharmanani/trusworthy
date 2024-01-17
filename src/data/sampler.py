import math
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset, WeightedRandomSampler
import torch.distributed as dist

T_co = TypeVar("T_co", covariant=True)


def metadata_to_gradelabel(metadata):
    import pandas as pd

    new_metadata = pd.DataFrame(metadata.copy())
    new_metadata[new_metadata["grade"] == "Benign"] = 0
    new_metadata[new_metadata["grade"] == "GS7"] = 1
    new_metadata[new_metadata["grade"] == "GS8"] = 2
    new_metadata[new_metadata["grade"] == "GS9"] = 2
    new_metadata[new_metadata["grade"] == "GS10"] = 2
    new_metadata["grade"]
    new_metadata["grade"] = new_metadata["grade"].astype(int)
    new_metadata["grade"]

    return new_metadata["grade"].to_numpy()


class WeightedDistributedSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: Dataset,
        targets,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:  # type:ignore
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)  # type:ignore
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas  # type:ignore
        self.shuffle = shuffle
        self.seed = seed
        self.targets = torch.tensor(targets).long()
        assert len(self.targets) == len(self.dataset)  # type:ignore

    def calculate_weights(self, targets):
        class_sample_count = torch.tensor(
            [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
        )
        weight = 1.0 / class_sample_count.double()
        samples_weight = torch.tensor([weight[t] for t in targets])
        return samples_weight

    def __iter__(self) -> Iterator[T_co]:

        indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        weights = list(self.calculate_weights(self.targets))

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)

        indices += indices[:padding_size]
        weights += weights[:padding_size]

        assert len(indices) == self.total_size
        assert len(weights) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        weights = weights[self.rank : self.total_size : self.num_replicas]

        assert len(indices) == self.num_samples
        assert len(weights) == self.num_samples

        g = torch.Generator()
        g.manual_seed(self.epoch + self.rank + self.seed)

        choices = torch.multinomial(
            torch.tensor(weights), self.num_samples, replacement=True, generator=g
        )

        indices = [indices[i] for i in choices]

        return iter(indices)  # type:ignore

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def get_sampling_weights_for_balanced_sampler(labels):
    import numpy as np

    labels = np.array(labels).astype(int)
    class_sample_count = np.array([(labels == t).sum() for t in np.unique(labels)])
    weights = 1.0 / class_sample_count
    samples_weight = np.array([weights[t] for t in labels])


def get_weighted_sampler(dataset, balance_grades=False):
    """_summary_

    Args:
        dataset (_type_): _description_
        balance_grades (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    Notes:
        This code can be used to create a weighted sampler for a dataset.
        The code can be reduced by combining the two if statements into one, but I left it for now.
    """

    if balance_grades:
        grade_labels = metadata_to_gradelabel(dataset.metadata)
        targets = torch.tensor(grade_labels).long()

        benign = torch.sum(targets == 0)
        lowGS = torch.sum(targets == 1)
        highGS = torch.sum(targets == 2)
        total = len(targets)

        benign_weight = total / benign
        lowGS_weight = total / lowGS
        highGS_weight = total / highGS

        # now we create a torch tensor with weights for each sample
        weights = torch.tensor(
            torch.where(
                targets == 0,
                benign_weight,
                torch.where(targets == 1, lowGS_weight, highGS_weight),
            )
        )

        return WeightedRandomSampler(
            weights,  # type:ignore
            len(weights),
        )

    targets = torch.tensor(dataset.labels).bool()

    positives = torch.sum(targets == True)
    negatives = torch.sum(targets == False)
    total = len(targets)
    positives_weight = total / positives
    negatives_weight = total / negatives

    # weights = torch.tensor(torch.where(targets, positives_weight, negatives_weight))
    # BUG alert
    weights = torch.where(targets, positives_weight, negatives_weight)

    return WeightedRandomSampler(
        weights,  # type:ignore
        len(weights),
    )  # type:ignore


def get_weighted_subset_sampler(dataset, labeled_indices, unlabeled_indices):
    """_summary_

    Args:
        dataset (_type_): _description_
        balance_grades (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_

    Notes:
        This code can be used to create a weighted sampler for a dataset.
        The code can be reduced by combining the two if statements into one, but I left it for now.
    """
    labels = torch.tensor(dataset.labels).bool()

    # labeled weights
    labeled_labels = labels[labeled_indices]
    labeled_set_targets = torch.tensor(labeled_labels).bool()

    labeled_set_positives = torch.sum(labeled_set_targets == True)
    labeled_set_negatives = torch.sum(labeled_set_targets == False)

    total = len(labeled_set_targets)
    positives_weight = total / labeled_set_positives
    negatives_weight = total / labeled_set_negatives
    # BUG alert
    labeled_set_weights = torch.where(
        labeled_set_targets, positives_weight, negatives_weight
    )

    # unlabeled weights
    unlabeled_labels = labels[unlabeled_indices]
    unlabeled_set_targets = torch.tensor(unlabeled_labels).bool()

    unlabeled_set_positives = torch.sum(unlabeled_set_targets == True)
    unlabeled_set_negatives = torch.sum(unlabeled_set_targets == False)

    total = len(unlabeled_set_targets)
    positives_weight = total / unlabeled_set_positives
    negatives_weight = total / unlabeled_set_negatives
    # BUG alert
    unlabeled_set_weights = torch.where(
        unlabeled_set_targets, positives_weight, negatives_weight
    )

    labeled_weighted_sampler = WeightedRandomSampler(
        labeled_set_weights,  # type:ignore
        len(labeled_set_weights),
    )  # type:ignore

    unlabeled_weighted_sampler = WeightedRandomSampler(
        unlabeled_set_weights,  # type:ignore
        len(unlabeled_set_weights),
    )  # type:ignore

    return labeled_weighted_sampler, unlabeled_weighted_sampler
