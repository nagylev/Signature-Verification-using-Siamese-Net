import os
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset, DataLoader

seed = 2020
np.random.seed(seed)


def data_loader(data, batch_size=12):
    data = SiameseDataSet(data)
    loader = DataLoader(data, batch_size=batch_size)
    return loader


class SiameseDataSet(Dataset):
    def __init__(self, data):
        self.df = data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        s1, s2, y = self.df.iloc[index]
#ey itt baj lehet
        s1 = Image.open(s1).convert('L')
        s2 = Image.open(s2).convert('L')

        return s1, s2, y
