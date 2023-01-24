import torch
from torch.utils.data import Dataset

from model.configs import PART_NUM

class Datasets(Dataset):
    def __init__(self, data, repeat=1):
        self.data = data * repeat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.data[idx][0][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.LongTensor([self.data[idx][1][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.LongTensor([self.data[idx][2][f"#{seq_idx}"] for seq_idx in range(PART_NUM)]),
            torch.FloatTensor(self.data[idx][0]["features"]),
            torch.LongTensor([self.data[idx][0]["label"]]),
        )
