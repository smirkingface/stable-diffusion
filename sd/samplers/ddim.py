import torch
import numpy as np
from tqdm import tqdm

from sd.samplers.sampler import Sampler

class DDIMSampler(Sampler):
    def __init__(self, num_timesteps, unconditional_guidance_scale=1, eta=0, dynamic_thresholding=None):
        super().__init__(unconditional_guidance_scale=unconditional_guidance_scale)
        self.num_timesteps = num_timesteps
        self.eta = eta
        self.alphas_cumprod = None
        
        self.dynamic_thresholding = dynamic_thresholding != None
        self.dynamic_thresholding_pars = {'percentile': 0.8, 'min_s': 1, 'clip_scale': 1, 'scale': 1}
        self.dynamic_thresholding_pars.update(dynamic_thresholding or {})
        # percentile: Determines s as given percentile
        # min_s: Minimum value of s (s = max(s, min_s))
        # clip_scale: Clips to [-s * clip_scale, s * clip_scale]
        # scale: Rescales to [-clip_scale/scale, clip_scale/scale]

    # Get DDIM parameters for the step from t to t_next (t=0 is endpoint, t=1000 is beginpoint, note that this is different than stored in alphas_cumprod (there t=0 is the penultimate step))
    def get_ddim_parameters(self, t, t_next):
        assert(t > 0 and t_next >= 0 and t_next < t) # Always go backwards through diffusion process
        
        # select alphas for computing the variance schedule
        alpha = self.alphas_cumprod[t-1]
        if t_next == 0:
            alpha_prev = 1
        else:
            alpha_prev = self.alphas_cumprod[t_next-1]

        # according the the formula provided in https://arxiv.org/abs/2010.02502
        sigma = self.eta * np.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))

        return alpha, alpha_prev, sigma

    def make_schedule(self, t_start=None, t_end=0, ddim_discretize='uniform'):
        t_start = t_start or self.model.schedule.num_timesteps
        self.alphas_cumprod = self.model.schedule.alphas_cumprod.cpu().numpy()
        
        # TODO: An equivalent method could be to calculate alphas_cumprod at the calculated ts and store that
        #       Allows option to calculate non-discrete steps
        
        if ddim_discretize == 'uniform':
            # Fixed DDIM steps, makes sure denoising process starts at num_ddpm_timesteps!
            self.ddim_timesteps = np.linspace(t_start, t_end, self.num_timesteps + 1).astype(int)
        elif ddim_discretize == 'quad':
            # TODO: Check and check how to implement t_start and t_end in this one
            self.ddim_timesteps = (np.linspace(np.sqrt(self.model.schedule.num_timesteps * .8), 0, self.num_timesteps + 1) ** 2).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discretize}"')

    def get_x_prev(self, x, e_t, t, t_next):
        a_t, a_prev, sigma_t = self.get_ddim_parameters(t, t_next)
        sqrt_one_minus_at = np.sqrt(1 - a_t)
        sqrt_at = np.sqrt(a_t)
    
        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / sqrt_at

        # Dynamic thresholding of pred_x0
        if self.dynamic_thresholding:
            b = pred_x0.shape[0]
            s = torch.quantile(abs(pred_x0.reshape(b,-1)), self.dynamic_thresholding_pars['percentile'], dim=1, keepdim=False).reshape(b,1,1,1)
            s = s.clip(self.dynamic_thresholding_pars['min_s'])
            pred_x0 = pred_x0.clip(-s * self.dynamic_thresholding_pars['clip_scale'], s * self.dynamic_thresholding_pars['clip_scale'])
            pred_x0 /= s * self.dynamic_thresholding_pars['scale']
            e_t = (x - sqrt_at*pred_x0) / sqrt_one_minus_at
        
        # direction pointing to x_t
        dir_xt = np.sqrt(1 - a_prev - sigma_t**2) * e_t
        noise = sigma_t * torch.randn_like(x)
        x_prev = np.sqrt(a_prev) * pred_x0 + dir_xt + noise
        return x_prev

    def p_sample(self, x, t, t_next, cond=None, unconditional_conditioning=None):
        e_t = self.get_model_output(x, t, cond, unconditional_conditioning)
        return self.get_x_prev(x, e_t, t, t_next)

    @torch.no_grad()
    def sample(self, shape=None, cond=None, unconditional_conditioning=None, mask=None, x0=None, x_init=None, t_start=None, t_end=0):
        device = self.model.device
        if x_init is None:
            x_init = torch.randn(shape, device=device)
        
        self.make_schedule(t_start, t_end)
        time_range = list(zip(self.ddim_timesteps[:-1], self.ddim_timesteps[1:]))
        
        print(f'Running DDIM Sampling with {len(time_range)} timesteps')
        iterator = tqdm(time_range, desc='DDIM Sampler', total=len(time_range))
    
        for i, (t, t_next) in enumerate(iterator):
            if mask != None:
                x_init = self.model.schedule.q_sample(x0, t) * mask + (1 - mask) * x_init
    
            x_init = self.p_sample(x_init, t, t_next, cond, unconditional_conditioning)

        if mask != None:
            x_init = x0 * mask + (1 - mask) * x_init

        return x_init
