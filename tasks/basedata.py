from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset
import torch


class BaseDataset(IterableDataset,ABC):
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError
    
    def _count_tokens(self, rows):
        raise NotImplementedError

    def estimate_total_token(self, sample_num=2000):
        # 采样 sample_num 条估平均长度,再乘总条数
        total_row = len(self)
        if total_row == 0:
            return 0
        sample_num = min(sample_num, total_row)
        row_loader = self._get_file_row(128, 0, 0, 0)
        token_sum = 0
        counted = 0
        while counted < sample_num:
            rows, *_ = next(row_loader)
            rows = rows[:sample_num - counted]
            token_sum += self._count_tokens(rows)
            counted += len(rows)
        return int(token_sum / counted * total_row)


class MixedDataset(BaseDataset):
    """按数据集行数比例交错多个具有相同 batch 协议的数据集。

    子数据集必须把自己的 state_dict 放在 yield 的最后一项。
    """

    def __init__(self, *datasets: BaseDataset, resume_state_dict=None):
        if not datasets:
            raise ValueError("MixedDataset requires at least one dataset")
        self.datasets = datasets
        self.total_dataset_num = len(datasets)
        self.dataset_len = [len(d) for d in datasets]
        if any(length <= 0 for length in self.dataset_len):
            raise ValueError(f"MixedDataset contains an empty dataset: {self.dataset_len}")
        total_len = float(sum(self.dataset_len))
        self.proportions = [length / total_len for length in self.dataset_len]
        saved_count = (resume_state_dict or {}).get('count')
        self.count = list(saved_count) if saved_count is not None else [0] * self.total_dataset_num
        if len(self.count) != self.total_dataset_num:
            raise ValueError("resume count does not match number of datasets")
        self.dataset_iter = [iter(d) for d in datasets]
        saved_children = (resume_state_dict or {}).get('children')
        self.child_state = list(saved_children) if saved_children is not None else [None] * self.total_dataset_num
        if len(self.child_state) != self.total_dataset_num:
            raise ValueError("resume children do not match number of datasets")

    def __len__(self):
        return sum(self.dataset_len)
    
    def state_dict(self):
        progress = 0.0
        for proportion, state in zip(self.proportions, self.child_state):
            if state:
                progress += proportion * float(state.get('progress', 0.0))
        return {
            'count': list(self.count),
            'children': list(self.child_state),
            'progress': progress,
        }
    
    def __iter__(self):
        while True:
            idx = min(range(self.total_dataset_num),
                      key=lambda i: self.count[i] / self.proportions[i])
            self.count[idx] += 1
            *data, child_sd = next(self.dataset_iter[idx])
            self.child_state[idx] = child_sd
            yield (*data, self.state_dict())

    def estimate_total_token(self, sample_num=2000):
        return sum(d.estimate_total_token(sample_num) for d in self.datasets)


class BatchShardDataset(BaseDataset):
    """Give each distributed rank a non-overlapping slice of every batch.

    The wrapped dataset must yield one or more tensors followed by its state
    dict.  It should be constructed with ``local_batch_size * world_size`` and
    on CPU, so only this rank's slice is copied to its CUDA device.
    """

    def __init__(self, dataset: BaseDataset, rank: int, world_size: int,
                 local_batch_size: int, device):
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} must be in [0, {world_size})")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.local_batch_size = local_batch_size
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        start = self.rank * self.local_batch_size
        stop = start + self.local_batch_size
        expected_batch_size = self.local_batch_size * self.world_size
        for *batch, state_dict in self.dataset:
            local_batch = []
            for value in batch:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "BatchShardDataset expects tensor batch fields, "
                        f"got {type(value).__name__}"
                    )
                if value.shape[0] != expected_batch_size:
                    raise ValueError(
                        "distributed SFT source returned batch dimension "
                        f"{value.shape[0]}, expected {expected_batch_size}"
                    )
                local_batch.append(
                    value[start:stop].contiguous().to(self.device, non_blocking=True)
                )
            yield (*local_batch, state_dict)

    def estimate_total_token(self, sample_num=2000):
        return self.dataset.estimate_total_token(sample_num)
