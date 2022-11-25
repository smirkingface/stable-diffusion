import torch
from tqdm import tqdm

from sd.samplers.ddim import DDIMSampler

class PLMSSampler(DDIMSampler):
    def __init__(self, num_timesteps, unconditional_guidance_scale=1, dynamic_thresholding=None):
        super().__init__(num_timesteps=num_timesteps, unconditional_guidance_scale=unconditional_guidance_scale, dynamic_thresholding=dynamic_thresholding)

    def p_sample(self, x, t, t_next, cond=None, unconditional_conditioning=None, old_eps=None):
        e_t,_ = self.get_model_output(x, t, cond, unconditional_conditioning)
        
        if len(old_eps) == 0:
            # Pseudo Improved Euler (2nd order)
            x_prev = self.get_x_prev(x, e_t, t, t_next)
            e_t_next,_ = self.get_model_output(x_prev, t_next, cond, unconditional_conditioning)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            # 2nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            # 3nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            # 4nd order Pseudo Linear Multistep (Adams-Bashforth)
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev = self.get_x_prev(x, e_t_prime, t, t_next)

        return x_prev, e_t

    @torch.no_grad()
    def sample(self, shape=None, cond=None, unconditional_conditioning=None, mask=None, x0=None, x_init=None, t_start=None, t_end=0):
        device = self.model.device
        if x_init is None:
            x_init = torch.randn(shape, device=device)
        
        self.make_schedule(t_start, t_end)
        time_range = list(zip(self.ddim_timesteps[:-1], self.ddim_timesteps[1:]))
        
        print(f'Running PLMS Sampling with {len(time_range)} timesteps')
        iterator = tqdm(time_range, desc='PLMS Sampler', total=len(time_range))
    
        old_eps = []
        for i, (t, t_next) in enumerate(iterator):
            if mask != None:
                x_init = self.model.schedule.q_sample(x0, t) * mask + (1 - mask) * x_init
    
            x_init, e_t = self.p_sample(x_init, t, t_next, cond, unconditional_conditioning, old_eps)
            
            old_eps.append(e_t)
            if len(old_eps) >= 4:
                old_eps.pop(0)
                
        if mask != None:
            x_init = x0 * mask + (1 - mask) * x_init

        return x_init
