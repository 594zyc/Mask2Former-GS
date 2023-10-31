import random
import math
import itertools
import random
from typing import Optional, Iterator, List, Union, Iterable

import torch
from torch.utils.data import (
    Dataset,
    DistributedSampler,
    Sampler,
    BatchSampler,
)
import torch.distributed as dist


# From: https://github.com/catalyst-team/catalyst/blob/2eee9d9cd7eb1e396fa9a4af7c5fadeeafbdaa38/Lib/operator.py#L271 # noqa
class itemgetter:
    """
    Return a callable object that fetches the given item(s) from its operand.
    After f = itemgetter(2), the call f(r) returns r[2].
    After g = itemgetter(2, 5, 3), the call g(r) returns (r[2], r[5], r[3])
    """

    __slots__ = ("_items", "_call")

    def __init__(self, item, *items):
        if not items:
            self._items = (item,)

            def func(obj):
                return (obj[item],)

            self._call = func
        else:
            self._items = items = (item,) + items

            def func(obj):
                return tuple(obj[i] for i in items)

            self._call = func

    def __call__(self, obj):
        return self._call(obj)

    def __repr__(self):
        return "%s.%s(%s)" % (
            self.__class__.__module__,
            self.__class__.__name__,
            ", ".join(map(repr, self._items)),
        )

    def __reduce__(self):
        return self.__class__, self._items


# From: https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/dataset.py#L6 # noqa
class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.

    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler) -> None:
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int) -> int:
        """Gets element of the dataset.

        Args:
            index: index of the element in the dataset

        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)


# From: https://github.com/catalyst-team/catalyst/blob/e99f90655d0efcf22559a46e928f0f98c9807ebf/catalyst/data/sampler.py#L499 # noqa
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.

    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.

    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        """

        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.

        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset

        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))


class MultiDatasetBatchSampler(BatchSampler):
    def __init__(
        self,
        datasets: Iterable[Dataset],
        batch_sizes: Union[int, Iterable[int]],
        sample_ratios: Optional[Iterable[float]] = None,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        """
        A DDP compatible batch sampler that samples from multiple datasets with
        different batch sizes and sample ratios.

        Args:
            datasets: an iterable of datasets
            batch_sizes: an iterable of batch sizes. If an int is provided, it will be
                used for all datasets. If a list is provided, it should have the same
                length as datasets.
            sample_ratios: an iterable of sample ratios. If None, all samples will be
                used. If provided, it should have the same length as datasets, and
                each ratio should be in (0, 1].
            drop_last: whether to drop the last batch if it is not full
            shuffle: whether to shuffle the samples and batches
        """

        assert len(datasets) > 0, "datasets should not be an empty iterable"

        if isinstance(batch_sizes, int):
            batch_sizes = [batch_sizes] * len(datasets)

        assert len(batch_sizes) == len(
            datasets
        ), "batch_sizes should have the same length as datasets"

        if sample_ratios is not None:
            assert len(sample_ratios) == len(
                datasets
            ), "sample_ratios should have the same length as datasets"
            assert all(
                0 < ratio and ratio <= 1 for ratio in sample_ratios
            ), "sample_ratios should be in (0, 1]"

        self.datasets = datasets
        self.offsets = [0] + list(itertools.accumulate(len(d) for d in self.datasets))
        self.batch_sizes = batch_sizes
        self.sample_ratios = (
            sample_ratios if sample_ratios is not None else [1] * len(datasets)
        )
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.is_ddp = dist.is_initialized()

        # calculate the overall length
        self.length = 0
        for dataset, batch_size, ratio in zip(
            self.datasets, self.batch_sizes, self.sample_ratios
        ):
            num_replica = dist.get_world_size() if self.is_ddp else 1
            length = math.ceil(int(len(dataset) * ratio) / num_replica)
            self.length += length // batch_size if drop_last else math.ceil(length / batch_size)

        self.epoch = 0
        self.seed = seed

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = []
        for dataset, batch_size, ratio, offset in zip(
            self.datasets, self.batch_sizes, self.sample_ratios, self.offsets
        ):
            
            subset_length = int(len(dataset) * ratio)
            if self.shuffle:
                random.seed(self.seed + self.epoch)
                indices = random.sample(range(offset, offset + len(dataset)), subset_length)
            else:
                indices = range(offset, offset + subset_length)
            
            if self.is_ddp:
                indices = DistributedSamplerWrapper(
                    indices, shuffle=self.shuffle, drop_last=self.drop_last
                )
            batch_sampler = BatchSampler(
                sampler=indices,
                batch_size=batch_size,
                drop_last=self.drop_last,
            )
            all_batches.extend(list(batch_sampler))

        # print(f"[rank {dist.get_rank()}] epoch {self.epoch}", all_batches)
        if self.shuffle:
            random.shuffle(all_batches)

        self.epoch += 1
        return iter(all_batches)

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def main():
    import os
    import time
    from torch.utils.data import ConcatDataset

    size = 10 # 1e6
    dummydataset1 = torch.utils.data.TensorDataset(torch.arange(size))
    dummydataset2 = torch.utils.data.TensorDataset(torch.arange(size*2) + 100)
    concat_dataset = ConcatDataset([dummydataset1, dummydataset2])

    if "RANK" in os.environ:
        print("Using DDP")
        dist.init_process_group("nccl")

    sampler = MultiDatasetBatchSampler(
        concat_dataset.datasets,
        batch_sizes=[1, 2],
        sample_ratios=[1, 0.7],
        shuffle=True,
        drop_last=False,
    )
    print(f"Length: {len(sampler)}")

    rank = dist.get_rank() if dist.is_initialized() else 0
    num_epochs = 2
    start_time = time.time()
    for e in range(num_epochs):
        # sampler.set_epoch(e)

        logstr = ""
        logstr += f"[rank {rank}] Epoch {e}\n"
        for batch in sampler:
            logstr += (
                f"[rank {rank}] {str(batch):<15} {[concat_dataset[b] for b in batch]}\n"
            )

        print(logstr)
        print(f"Time taken: {time.time() - start_time:.2f}s")
        os.system("sleep 1")

    if "RANK" in os.environ:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
