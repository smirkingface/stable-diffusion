import argparse, os, sys
import yaml
from torch.utils.data import IterableDataset
from PIL import Image
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
        help='Configuration filename'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=23,
        help='seed for seed_everything',
    )
    parser.add_argument(
        '-n',
        type=int,
        default=100,
        help='Maximum number of training samples to draw from training dataloader',
    )

    return parser

if __name__ == '__main__':
    # add cwd for convenience and to make classes in this file available when running as `python test_dataloader.py`
    sys.path.append(os.getcwd())

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
        
    # Test training dataloader, put results in ./test_dataloader
    train_dataloader = data.train_dataloader()
    
    os.makedirs('./test_dataloader', exist_ok=True)
    it = iter(train_dataloader)
    with open('./test_dataloader/captions.txt', 'w') as fp:
        for i in range(opt.n):
            x = next(it)
            fp.write(str(x['caption']) + '\n')
            Image.fromarray(((x['image'].reshape(x['image'].shape[0]*x['image'].shape[1], x['image'].shape[2], x['image'].shape[3]).cpu().numpy() + 1) * 127.5).astype(np.uint8)).save(f'./test_dataloader/train_sample_{i}.png')
            if 'mask' in x:
                Image.fromarray(((x['mask'].reshape(x['mask'].shape[0]*x['mask'].shape[1], x['mask'].shape[2]).cpu().numpy()) * 255.0).astype(np.uint8), 'L').save(f'./test_dataloader/train_sample_{i}_mask.png')
