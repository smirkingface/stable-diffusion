import argparse, sys, os
sys.path.append(os.getcwd())

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
import yaml

from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext

from sd.util import load_model_from_config
from sd.samplers.ddpm import DDPMSampler
from sd.samplers.ddim import DDIMSampler
from sd.samplers.plms import PLMSSampler


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default='a painting of a virus monster playing guitar',
        help='the prompt to render'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        nargs='?',
        help='dir to write results to',
        default='outputs/txt2img-samples'
    )
    parser.add_argument(
        '--skip_grid',
        action='store_true',
        help='do not save a grid, only individual samples. Helpful when evaluating lots of samples',
    )
    parser.add_argument(
        '--skip_save',
        action='store_true',
        help='do not save indiviual samples. For speed measurements.',
    )
    parser.add_argument(
        '--steps',
        type=int,
        default=50,
        help='number of ddim sampling steps',
    )
    parser.add_argument(
        '--ddim_eta',
        type=float,
        default=1.0,
        help='ddim eta (eta=0.0 corresponds to deterministic sampling)',
    )
    parser.add_argument(
        '--n_iter',
        type=int,
        default=1,
        help='sample this often',
    )
    parser.add_argument(
        '--H',
        type=int,
        default=512,
        help='image height, in pixel space',
    )
    parser.add_argument(
        '--W',
        type=int,
        default=512,
        help='image width, in pixel space',
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=4,
        help='how many samples to produce for each given prompt. A.k.a batch size',
    )
    parser.add_argument(
        '--n_rows',
        type=int,
        default=0,
        help='rows in the grid (default: n_samples)',
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=5.0,
        help='unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))',
    )
    parser.add_argument(
        '--from-file',
        type=str,
        help='if specified, load prompts from this file',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='path to config which constructs model',
    )
    parser.add_argument(
        '--ckpt',
        type=str,
        default='./models/model.ckpt',
        help='path to checkpoint of model',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='the seed (for reproducible sampling)',
    )
    parser.add_argument(
        '--precision',
        type=str,
        help='evaluate at this precision',
        choices=['full', 'autocast'],
        default='autocast'
    )
    parser.add_argument(
        '--sampler',
        type=str,
        help='sampler type to use',
        choices=['ddpm', 'ddim', 'plms'],
        default='ddim'
    )
    parser.add_argument(
        '--use_ema',
        action='store_true',
        help='Use EMA weights',
    )
    parser.add_argument(
        '--same_seed',
        action='store_true',
        help='Use same seed for every prompt',
    )
    parser.add_argument(
        '--v2',
        action='store_true',
        help='Use SD 2.0 model architecture',
    )
    opt = parser.parse_args()

    if opt.config:
        config = yaml.safe_load(open(opt.config, 'r'))
    else:
        if opt.v2:
            config = {'model': {'target': 'sd.models.diffusion.StableDiffusionV2'}}
        else:
            config = {'model': {'target': 'sd.models.diffusion.StableDiffusion'}}
    model = load_model_from_config(config, opt.ckpt, verbose=True, swap_ema=opt.use_ema, no_ema=not opt.use_ema)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    if opt.sampler == 'ddpm':
        sampler = DDPMSampler(num_timesteps=opt.steps)
    elif opt.sampler == 'ddim':
        sampler = DDIMSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta)
    elif opt.sampler == 'plms':
        sampler = PLMSSampler(num_timesteps=opt.steps, unconditional_guidance_scale=opt.scale, eta=opt.ddim_eta)
    else:
        raise ValueError(f'Unknown sampler type {opt.sampler}')

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        data = [batch_size * [prompt]]
    else:
        print(f'Reading prompts from {opt.from_file}')
        with open(opt.from_file, 'r') as f:
            data = f.read().splitlines()
            data = [batch_size * [prompt] for prompt in data]

    sample_path = os.path.join(outpath, 'samples')
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    seed_everything(opt.seed)

    precision_scope = autocast if opt.precision=='autocast' else nullcontext
    with precision_scope('cuda'):
        tic = time.time()
        for prompts in tqdm(data, desc='data'):
            if opt.same_seed:
                seed_everything(opt.seed)
                
            all_samples = list()
            for n in trange(opt.n_iter, desc='Sampling'):
                shape = [opt.H, opt.W]
                x_samples = model.sample(prompts, sampler, shape=shape)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0).cpu()

                if not opt.skip_save:
                    for x_sample in x_samples:
                        x_sample = 255 * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f'{base_count:05}.png'))
                        base_count += 1
                
                if not opt.skip_grid:
                    all_samples.append(x_samples)

            if not opt.skip_grid:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255 * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                grid_count += 1

        toc = time.time()

    print(f'Your samples are ready and waiting for you here: \n{outpath} \n'
          f'Sampling took {toc - tic}s, i.e. produced {opt.n_iter * opt.n_samples / (toc - tic):.2f} samples/sec.'
          f' \nEnjoy.')


if __name__ == '__main__':
    main()
