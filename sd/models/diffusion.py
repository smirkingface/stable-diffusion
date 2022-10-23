import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from einops import rearrange
from contextlib import contextmanager
from copy import deepcopy

from sd.util import count_params, instantiate_from_config
from sd.modules.ema import LitEma
from sd.modules.distributions import DiagonalGaussianDistribution
from sd.modules.schedule import DDPMSchedule

class StableDiffusion(pl.LightningModule):
    def __init__(self, unet_config={'target': 'sd.modules.unet.UNetModel'},
                 first_stage_config={'target': 'sd.models.autoencoder.AutoencoderKL'},
                 cond_stage_config={'target': 'sd.modules.clip.FrozenCLIPEmbedder'},
                 cond_stage_key='caption',
                 cond_stage_trainable=False,
                 conditioning_key='crossattn',
                 base_learning_rate=1e-6,
                 scale_factor=0.18215,
                 loss_type='l2',
                 use_ema=True,
                 reset_ema=False,
                 first_stage_key='image',
                 mask_key='mask', # By default, use a training mask if provided by the data loader. Disable by setting the key to None
                 channels=4,
                 original_elbo_weight=0.,
                 l_simple_weight=1.,
                 schedule_parameters={},
                 scheduler_config=None,
                 optimizer_config=None,
                 sd_compatibility=True
                 ):
        super().__init__()
        self.cond_stage_model = None
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.mask_key = mask_key
        self.channels = channels
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        
        self.use_ema = use_ema
        self.reset_ema = reset_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f'Keeping EMAs of {len(list(self.model_ema.buffers()))}.')

        self.learning_rate = base_learning_rate
        self.scheduler_config = scheduler_config
        self.optimizer_config = optimizer_config

        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.schedule = DDPMSchedule(**schedule_parameters)
        self.loss_type = loss_type
         
        self.cond_stage_trainable = cond_stage_trainable
        self.scale_factor = scale_factor
        self.sd_compatibility = sd_compatibility

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('global_step', float(self.global_step), prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        if self.use_ema:
            with self.ema_scope():
                _, loss_dict_ema = self.shared_step(batch)
                loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)


    # txt2img: Supply shape
    # img2img: Supply x0 and t_start
    # Masking for either: Supply x0 and mask
    # Fixed starting noise: Supply x_init
    # Note: shape is ignored if x0 or x_init is given
    def sample(self, prompts, sampler, shape=None, x0=None, x_init=None, mask=None, t_start=None, t_end=0, cfg=True, cfg_prompt='', skip_decode=False):
        sampler.set_model(self)
        
        c = self.get_learned_conditioning(prompts)
        uc = None
        if cfg:
            uc = self.get_learned_conditioning(len(prompts) * [cfg_prompt])
        
        if x0 != None:
            t_start = t_start or self.schedule.num_timesteps
            x0 = self.encode_first_stage(x0)

            if t_start != None:
                x_init = self.schedule.q_sample(x_start=x0, t=t_start-1, noise=x_init)

        if shape != None:
            downsample = 2**(self.first_stage_model.encoder.num_resolutions-1)
            shape = [len(prompts), self.first_stage_model.z_channels, shape[0] // downsample, shape[1] // downsample]
        
        if mask != None:
            assert(x0 != None)
            # Downsample mask if not supplied in latent space format
            if mask.shape[2:] != x0.shape[2:]:
                mask = nn.functional.avg_pool2d(mask, list(np.int16(np.array(mask.shape[2:])/x0.shape[2:])))
            assert(mask.shape == x0.shape)
        
        x_samples = sampler.sample(shape=shape, cond=c, unconditional_conditioning=uc, x_init=x_init, mask=mask, t_start=t_start, t_end=t_end)

        if not skip_decode:
            x_samples = self.decode_first_stage(x_samples)
        
        return x_samples

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = lambda self,mode=None: self # Overwrite train() to make sure the model does not get set to train mode
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        model = instantiate_from_config(config)
        self.cond_stage_model = model
        
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model.eval()
            self.cond_stage_model.train = lambda self,mode=None: self # Overwrite train() to make sure the model does not get set to train mode
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
    
    def get_learned_conditioning(self, c):
        if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
            c = self.cond_stage_model.encode(c)
            if isinstance(c, DiagonalGaussianDistribution):
                c = c.mode()
        else:
            c = self.cond_stage_model(c)

        return c

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.swap(self.model)
            if context is not None:
                print(f'{context}: Switched to EMA weights')
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.swap(self.model)
                if context is not None:
                    print(f'{context}: Restored training weights')
                    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            return torch.nn.functional.l1_loss(pred, target, reduction='mean' if mean else 'none')
        elif self.loss_type == 'l2':
            return torch.nn.functional.mse_loss(pred, target, reduction='mean' if mean else 'none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x):
        encoder_posterior = self.first_stage_model.encode(x)
        
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def shared_step(self, batch, **kwargs):
        x = self.get_input(batch, self.first_stage_key).to(self.device)
        x = self.encode_first_stage(x)
        c = batch[self.cond_stage_key]
        
        mask = None
        if self.mask_key is not None and self.mask_key in batch:
            mask = self.get_input(batch, self.mask_key).to(self.device)
            # Downsample image-space mask to latent space
            mask = nn.functional.avg_pool2d(mask, list(np.int16(np.array(mask.shape[2:])/x.shape[2:])))

        loss = self(x, c, mask)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.schedule.num_timesteps, (x.shape[0],), device=self.device).long()
        # TODO: Is this code in the right place?
        if isinstance(c, dict) or isinstance(c, list):
            c = self.get_learned_conditioning(c)
        else:
            c = self.get_learned_conditioning(c.to(self.device))
        return self.p_losses(x, c, t, *args, **kwargs)

    def apply_model(self, x_noisy, t, cond):
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond)
        return x_recon

    def p_losses(self, x_start, cond, t, mask=None, noise=None):
        if mask == None:
            mask = 1
        
        if noise is None:
            noise = torch.randn_like(x_start)
        x_noisy = self.schedule.q_sample(x_start=x_start*mask, t=t, noise=noise*mask)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        loss_simple = self.get_loss(model_output*mask, noise*mask, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        loss = self.l_simple_weight * loss_simple.mean()

        # TODO: lvlb_weights in schedule, or in this module?
        if self.original_elbo_weight > 0:
            loss_vlb = (self.schedule.lvlb_weights[t] * loss_simple).mean()
            loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
            loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())

        # Keep only parameters that require gradients
        params = [x for x in params if x.requires_grad]
        
        if self.optimizer_config != None:
            config = deepcopy(self.optimizer_config)
            if 'params' not in config:
                config['params'] = {}
            config['params']['params'] = params
            config['params']['lr'] = lr
            opt = instantiate_from_config(config)
        else:
            opt = torch.optim.AdamW(params, lr=lr)
        
        if self.scheduler_config != None:
            assert 'scheduler' in self.scheduler_config
            config = deepcopy(self.scheduler_config)
            if 'params' not in config['scheduler']:
                config['scheduler']['params'] = {}
            config['scheduler']['params']['optimizer'] = opt
            config['scheduler'] = instantiate_from_config(config['scheduler'])
            return {'optimizer': opt, 'lr_scheduler': config}
        return opt
    
    # Overloaded load_state_dict for providing advanced logic regarding EMA weights and compatibility with SD
    def load_state_dict(self, state_dict, strict=True):
        reset_ema = False
        if not self.use_ema:
            # Remove model_ema fields from state_dict
            k = list(state_dict.keys())
            for x in k:
                if x.startswith('model_ema.'):
                    del state_dict[x]
        else:
            # Check if all parameters are in the model_ema state_dict
            for key,v in self.model.named_parameters():
                if v.requires_grad:
                    k = self.model_ema.m_name2s_name[key]
                    if f'model_ema.{k}' not in state_dict:
                        print(f'Not all model parameters were found in the model_ema state_dict (\'{key}\' missing), resetting!')
                        reset_ema = True
                        break
            
            if not reset_ema:
                # Check if any model_ema state_dict parameters are not in model parameters
                k = list(state_dict.keys())
                for x in k:
                    if x.startswith('model_ema.'):
                        if x[10:] in ['decay', 'num_updates']:
                            continue
                        
                        if x[10:] not in self.model_ema.m_name2s_name.values():
                            print(f'{x} in model_ema state_dict, but not in model, removing!')
                            del state_dict[x]

        if self.use_ema and (self.reset_ema or reset_ema):
            # Remove model_ema fields from state_dict (get rid of any extra model_ema parameters)
            k = list(state_dict.keys())
            for x in k:
                if x.startswith('model_ema.'):
                    del state_dict[x]
                    
            # Copy current parameters into model_ema state dict
            for key,v in self.model.named_parameters():
                if v.requires_grad:
                    k = self.model_ema.m_name2s_name[key]
                    state_dict[f'model_ema.{k}'] = v.data
            state_dict['model_ema.num_updates'] = torch.tensor(0, dtype=torch.int)
        
        if self.sd_compatibility:
            # Remove DDPM schedule keys from state_dict (these are calculated in DDPMScheduler)
            schedule_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']
            for x in schedule_keys:
                if x in state_dict:
                    del state_dict[x]
        
        return super().load_state_dict(state_dict, strict=strict)
    
    # Overloaded state_dict for providing compatibility of the new DDPMScheduler with original SD checkpoints
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        state_dict = super().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
        
        if self.sd_compatibility:
            # Fetch DDPM schedule keys and put them into state_dict
            schedule_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']
            for x in schedule_keys:
                state_dict[f'{prefix}{x}'] = getattr(self.schedule, x)
            
            # Recreate log_one_minus_alphas_cumprod (does not really matter, it is unused)
            state_dict[f'{prefix}log_one_minus_alphas_cumprod'] = torch.log(1 - state_dict[f'{prefix}alphas_cumprod'])
        return state_dict


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid-adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None, c_adm=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
