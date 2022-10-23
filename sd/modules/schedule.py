import torch
import numpy as np
from functools import partial

from sd.modules.util import extract_into_tensor

# TODO: Fix arrays bouncing between torch and numpy

class DDPMSchedule(torch.nn.Module):
    def __init__(self, timesteps=1000, beta_schedule='quad', linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3, v_posterior=0):
        super().__init__()
        self.register_schedule(beta_schedule=beta_schedule, timesteps=timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s, v_posterior=v_posterior)

    def make_beta_schedule(self, schedule, n_timestep, linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3):
        if schedule == 'quad':
            betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        elif schedule == 'linear':
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
        elif schedule == 'cosine':
            timesteps = torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s

            alphas = timesteps / (1 + cosine_s) * np.pi / 2
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas, a_min=0, a_max=0.999)
        elif schedule == 'sqrt':
            betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
        else:
            raise ValueError(f"schedule '{schedule}' unknown.")
        return betas.numpy()

    def register_schedule(self, beta_schedule='quad', timesteps=1000, linear_start=0.00085, linear_end=0.0120, cosine_s=8e-3, v_posterior=0):
        betas = self.make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        
        alphas = 1 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1, alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas), persistent=False) # Unused
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod), persistent=False)
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev), persistent=False)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1 - alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1 / alphas_cumprod)), persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1 / alphas_cumprod - 1)), persistent=False)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('posterior_variance', to_torch((1 - v_posterior) * betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod) + v_posterior * betas), persistent=False)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', torch.log(self.posterior_variance.clamp(min=1e-20)), persistent=False)
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)), persistent=False)
        self.register_buffer('posterior_mean_coef2', to_torch((1 - alphas_cumprod_prev) * np.sqrt(alphas) / (1 - alphas_cumprod)), persistent=False)

        lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
    
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not torch.isnan(self.lvlb_weights).all()
        
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_variance_clipped
