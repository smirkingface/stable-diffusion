import os
import math
import numpy as np
from PIL import Image

import torch
import torchvision

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.seed import isolate_rng

from sd.util import instantiate_from_config
from sd.samplers.ddim import DDIMSampler

class ImageLogger(Callback):
    def __init__(self, batch_size=1, shape=[512,512], prompts=[''], sampler_config=None, prefix='', every_n_steps=None, every_n_epochs=1, seed=None, grid=True, single_grid=False, use_ema=True):
        self.prompts = prompts
        self.first = True
        self.shape = shape
        self.batch_size = batch_size
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.seed = seed
        if sampler_config == None:
            self.sampler = DDIMSampler(50, unconditional_guidance_scale=5.0, eta=1.0)
        else:
            self.sampler = instantiate_from_config(sampler_config)
        self.prefix = prefix
        self.grid = grid
        self.single_grid = single_grid
        self.use_ema = use_ema
        self.last_step = -1

    def write_images(self, images, batch, logdir, epoch, global_step):
        root = os.path.join(logdir, 'images', 'prompts')
        for k in images:
            if self.grid:
                grid = torchvision.utils.make_grid(images[k], nrow=4)
                grid = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                filename = f'{self.prefix}{k}_gs-{global_step:06}_e-{epoch:06}_b{batch:06}.png'
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                Image.fromarray(grid).save(path)
            else:
                for i in range(images[k].shape[0]):
                    image = (images[k][i].permute(1,2,0).numpy() * 255).astype(np.uint8)
                    image_index = batch*self.batch_size + i
                    filename = f'{self.prefix}{k}_gs-{global_step:06}_e-{epoch:06}_b{image_index:06}.png'
                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    Image.fromarray(image).save(path)

    def sample_images(self, model, prompts):
        images = {}

        if self.use_ema and model.use_ema:
            # Reset RNG after sampling
            with isolate_rng():
                if self.seed != None:
                    torch.manual_seed(self.seed)
                with model.ema_scope("Plotting"):
                    x = model.sample(prompts, self.sampler, shape=self.shape)
                    images['samples_ema'] = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0).cpu()

        # Reset RNG after sampling
        with isolate_rng():
            if self.seed != None:
                torch.manual_seed(self.seed)
            x = model.sample(prompts, self.sampler, shape=self.shape)
            images['samples'] = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0).cpu()
        return images

    def log_images(self, trainer, pl_module):
        gr = trainer.global_rank
        ws = trainer.world_size
        logdir = trainer.log_dir

        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        image_set = None
        for b in range(int(math.ceil(len(self.prompts) / self.batch_size))):
            # Skip batches based on the rank of the current process (splits work across GPUs/nodes)
            if b % ws != gr:
                continue

            images = self.sample_images(pl_module, self.prompts[b*self.batch_size:(b+1)*self.batch_size])

            # TODO: Fix single_grid in combination with DDP training, all_gather the images to rank 0 and save from there.
            if self.single_grid:
                if image_set == None:
                    image_set = images
                else:
                    for k in image_set:
                        image_set[k] = torch.cat((image_set[k], images[k]), dim=0)
            else:
                self.write_images(images, b, logdir, pl_module.current_epoch, pl_module.global_step)

        if self.single_grid:
            self.write_images(image_set, 0, logdir, pl_module.current_epoch, pl_module.global_step)

        if is_train:
            pl_module.train()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_index):
        if not self.first and (self.every_n_steps == None or pl_module.global_step % self.every_n_steps != 0 or pl_module.global_step == self.last_step):
            return

        self.last_step = pl_module.global_step
        self.first = False
        self.log_images(trainer, pl_module)

    def on_train_epoch_start(self, trainer, pl_module):
        if not self.first and (self.every_n_epochs == None or pl_module.current_epoch % self.every_n_epochs != 0):
            return

        self.first = False
        self.log_images(trainer, pl_module)
