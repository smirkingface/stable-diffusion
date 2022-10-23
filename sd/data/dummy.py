import numpy as np
import random
import string
from torch.utils.data import Dataset

class DummyData(Dataset):
    def __init__(self, length, size):
        self.length = length
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = np.random.randn(*self.size)
        y = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
        return {"image": x, "caption": y}


class DummyDataWithEmbeddings(Dataset):
    def __init__(self, length, size, emb_size):
        self.length = length
        self.size = size
        self.emb_size = emb_size

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        x = np.random.randn(*self.size)
        y = np.random.randn(*self.emb_size).astype(np.float32)
        return {"image": x, "caption": y}

