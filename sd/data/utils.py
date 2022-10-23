import random
from functools import partial

from torch.utils.data import DataLoader, IterableDataset
import pytorch_lightning as pl

from sd.util import instantiate_from_config

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 num_workers=None, shuffle_test_loader=False, shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2

        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

    def setup(self, stage=None):
        self.datasets = {k:instantiate_from_config(v) for k,v in self.dataset_configs.items()}

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not is_iterable_dataset)

    def _val_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        shuffle = shuffle and (not is_iterable_dataset)
        return DataLoader(self.datasets["validation"],  batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], IterableDataset)
        shuffle = shuffle and (not is_iterable_dataset)
        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,  num_workers=self.num_workers)


# Abstract class for transforms that use additional information about the data
class AdvancedTransform:
    def __call__(self, data, info={}):
        return data, info

# Load a list of transforms from config
def load_transforms(transforms):
    return [instantiate_from_config(x) for x in transforms]

# Apply a list of transforms to data
def apply_transforms(transforms, data, info={}):
    for t in transforms:
        if isinstance(t, AdvancedTransform):
            data, info = t(data, info)
        else:
            data = t(data)
    return data


# Iterates over a randomly shuffled dataset
def shuffle_iterator(dataset):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    for i in indices:
        yield dataset[i]

# Iterable dataset that keeps buffers of items with the same shape (i.e. can be stacked together in a batch).
# Yields a series of images once a buffer reaches the desired batch_size (which DataLoader will stack into a batch)
# Stops after a predefined number of batches
class VariableShapeDataset(IterableDataset):
    def __init__(self, dataset, batch_size=1, total_batches=1):
        assert total_batches>0
        self.dataset = instantiate_from_config(dataset)
        self.total_batches = total_batches
        self.batch_size = batch_size
        self.buffer = {}

    def __iter__(self):
        i = 0
        while True:
            for it in (iter(self.dataset) if isinstance(self.dataset, IterableDataset) else shuffle_iterator(self.dataset)):
                sh = it['image'].shape[:2]
                if sh not in self.buffer:
                    self.buffer[sh] = []
                self.buffer[sh].append(it)
                
                if len(self.buffer[sh]) == self.batch_size:
                    yield from self.buffer[sh]
                    self.buffer[sh].clear()
                    i += 1
                    if i == self.total_batches:
                        return

# Alternates samples from multiple datasets with an optional limit on the number of items produced per pass
class AlternatingDataset(IterableDataset):
    def __init__(self, datasets, total_items=None):
        self.datasets = [instantiate_from_config(x) for x in datasets]
        self.total_items = total_items

    def __iter__(self):
        i = 0
        for items in zip(*[iter(x) if isinstance(x, IterableDataset) else shuffle_iterator(x) for x in self.datasets]):
            yield from items
            i += 1
            if i == self.total_items:
                break

# Repeats a dataset multiple times. Use to arbitrarily increase epoch length
class RepeatedDataset(IterableDataset):
    def __init__(self, dataset, repeats=1):
        self.dataset = instantiate_from_config(dataset)
        self.repeats = repeats
    
    def __iter__(self):
        for i in range(self.repeats):
            yield from (iter(self.dataset) if isinstance(self.dataset, IterableDataset) else shuffle_iterator(self.dataset))
