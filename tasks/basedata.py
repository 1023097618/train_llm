from abc import ABC, abstractmethod
from torch.utils.data import IterableDataset


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
