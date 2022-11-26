import argparse, os, sys
import yaml
import re

sys.path.insert(1, os.path.join(sys.path[0], '..'))  # be able to load sd.util from scripts/

from torch.utils.data import IterableDataset
from PIL import Image
from tqdm import tqdm
import numpy as np

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from sd.util import instantiate_from_config


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '-c',
        '--config',
        help='Configuration filename',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='tags.csv',
        help='Output CSV filename',
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-l',
        type=int,
        default=1,
        help='Number of training samples loops to use from training dataloader (dataset_samples * l = ttl_samples)',
    )
    parser.add_argument(
        '--tag-separator',
        default=',',
        help='Tag separator character',
    )
    parser.add_argument(
        '--split-words',
        help='Comma-delimited list of conjunction/preposition words to further split tags'
    )

    return parser

if __name__ == '__main__':

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt = parser.parse_args()
    seed_everything(opt.seed, workers=True)

    # Load config
    config = yaml.safe_load(open(opt.config, 'r'))

    # Create data
    data = instantiate_from_config(config['data'])
    data.prepare_data()
    data.setup()

    # Test validation dataloader if it exists
    try:
        val_dataloader = iter(data.val_dataloader())
        i = 0
        for x in val_dataloader:
            print(i)
            i+=1
        print(f'{i} items in validation dataloader')
        del val_dataloader
    except MisconfigurationException:
        pass

    # Run through the training DataLoader
    train_dataloader = data.train_dataloader()

    # Capture all of the tags and their weights
    loop = 1
    samples_ttl = len(train_dataloader) * opt.l
    tag_sums = {}
    it = iter(train_dataloader)

    # Compile complete list of tag splitters
    splitters = [re.escape(opt.tag_separator)]
    if opt.split_words:
        words = re.split(r'\s*,\s*', opt.split_words)
        words = [r'\b' + re.escape(w) + r'\b' for w in words]
        splitters.extend(words)

    tag_splitter = re.compile(r'\s*(?:' + '|'.join(splitters) + r')\s*')

    for i in tqdm (range(samples_ttl), desc="Parsing captions from samples"):
        try:
            x = next(it)
        except StopIteration:
            print(f'\nExhausted samples for Loop {loop}; re-running iterator')
            it = iter(train_dataloader)
            x = next(it)
            loop += 1
            pass

        tag_list = tag_splitter.split(x['caption'][0])
        for tag in tag_list:
            if tag in tag_sums:
                tag_sums[tag] += 1
            else:
                tag_sums[tag] = 1

    # Reverse sort by weight
    tag_sums = dict( sorted(tag_sums.items(), key=lambda x:x[1], reverse=True) )

    # Write the CSV file
    with open(opt.output, 'w') as fp:
        fp.write('Tag,Weight\n')
        for tag in tag_sums.keys():
            tag_str = f'"{tag}"' if tag.find(',') != -1 else tag
            weight_decimal = tag_sums[tag] / samples_ttl
            fp.write(f'{tag_str},{weight_decimal}\n')
