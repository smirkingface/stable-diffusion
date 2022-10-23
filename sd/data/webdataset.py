import webdataset as wds
import os
import numpy as np
import pytorch_lightning as pl
import torch
from functools import partial

from sd.data.utils import load_transforms, apply_transforms


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result

def apply_transforms_to_dict(data, transforms, key, info_key):
    data[key] = apply_transforms(transforms, data[key], info=data.get(info_key, None))
    return data

def convert_image(data):
    image = data['image']
    
    if image.mode == 'RGBA':
        image = np.array(image).astype(np.uint8)
        data['image'] = (image[...,:3] / 127.5 - 1.0).astype(np.float32)
        data['mask'] = (image[...,[3]] / 255.0).astype(np.float32)
    else:
        image = np.array(image).astype(np.uint8)
        data['image'] = (image / 127.5 - 1.0).astype(np.float32)
    
    keep_keys = ['image', 'mask', 'caption', '__key__'] # Keep __key__ because wds will reinsert it as None
    keys = list(data.keys())
    for k in keys:
        if k not in keep_keys:
            del data[k]

    return data

def compose_alpha(data, mask_key):
    if mask_key not in data:
        return data

    alpha = data[mask_key].convert('L')
    data['image'].putalpha(alpha)
    del data[mask_key]
    return data

# TODO: Maybe add options for resampled and with_epoch
def WebDataset(tar_base, shards, shuffle=True, shuffle_len=1000, multinode=True, min_size=None,
               transforms=[], caption_transforms=[], mask_key=None):
    image_transforms = load_transforms(transforms)
    caption_transforms = load_transforms(caption_transforms)

    nodesplitter = wds.shardlists.split_by_node if multinode else wds.shardlists.single_node_only
    tars = os.path.join(tar_base, shards)

    dset = wds.WebDataset(tars, nodesplitter=nodesplitter, shardshuffle=shuffle)
    if shuffle:
        dset = dset.shuffle(shuffle_len)
    print(f'Loading WebDataset with {len(dset.pipeline[0].urls)} shards.')

    image_formats = ['jpg', 'jpeg', 'png']
    if mask_key:
        image_formats = filter(lambda x: x != mask_key, image_formats)

    dset = (dset
            .select(lambda x: ('jpg' in x or 'jpeg' in x or 'png' in x) and ('txt' in x))
            .decode('pil')
            .rename(image=';'.join(image_formats), caption='txt')
            )

    if mask_key:
        dset = dset.map(partial(compose_alpha, mask_key=mask_key))

    def filter_size(x):
        try:
            valid = True
            if min_size is not None and min_size > 1:
                try:
                    valid = valid and x['json']['original_width'] >= min_size and x['json']['original_height'] >= min_size
                except Exception:
                    valid = False
            return valid
        except Exception:
            return False

    dset = (dset
            .select(filter_size)
            .map(partial(apply_transforms_to_dict, transforms=image_transforms, key='image', info_key='json'))
            .map(partial(apply_transforms_to_dict, transforms=caption_transforms, key='caption', info_key='json'))
            .map(convert_image)
            )
    
    return dset

class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, multinode=True):
        super().__init__()
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode

    def make_loader(self, dataset_config, shuffle=True):
        # TODO: Is there any benefit using wds batched(...) over the default pytorch batch collation?
        dset = WebDataset(self.tar_base, shuffle=shuffle, multinode=self.multinode, **dataset_config).batched(self.batch_size, partial=False, collation_fn=dict_collation_fn)
        loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=self.num_workers)
        return loader

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, shuffle=False)

    def test_dataloader(self):
        return self.make_loader(self.test, shuffle=False)


    
