import torch
import numpy as np
from tqdm import tqdm

from sd.samplers.sampler import Sampler

class DDPMSampler(Sampler):
    def __init__(self, num_timesteps, unconditional_guidance_scale=1, clip_denoised=False):
        super().__init__(unconditional_guidance_scale=unconditional_guidance_scale)
        self.num_timesteps = num_timesteps
        self.clip_denoised = clip_denoised
    
    def p_mean_variance(self, x, t, cond=None, unconditional_conditioning=None):
        model_out = self.get_model_output(x, t, cond, unconditional_conditioning)
        x_recon = self.model.schedule.predict_start_from_noise(x, t, model_out)

        if self.clip_denoised:
            x_recon.clamp_(-1, 1)

        return self.model.schedule.q_posterior(x_recon, x, t)

    def p_sample(self, x, t, cond=None, unconditional_conditioning=None):
        b, *_, device = *x.shape, x.device
        model_mean, model_log_variance = self.p_mean_variance(x, t, cond, unconditional_conditioning)
        noise = torch.randn(x.shape, device=device)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def sample(self, shape=None, cond=None, unconditional_conditioning=None, mask=None, x0=None, x_init=None, t_start=None, t_end=0):
        device = self.model.device
        if x_init is None:
            x_init = torch.randn(shape, device=device)
        
        print(f'Running DDPM Sampling with {self.num_timesteps} timesteps')
        iterator = tqdm(np.linspace(t_start, t_end, self.num_timesteps).astype(int), desc='DDPM Sampler', total=self.num_timesteps)

        for t in iterator:            
            if mask != None:
                x_init = self.model.schedule.q_sample(x0, t) * mask + (1 - mask) * x_init
            
            x_init = self.p_sample(x_init, t, cond, unconditional_conditioning)

        if mask != None:
            x_init = x0 * mask + (1 - mask) * x_init

        return x_init
