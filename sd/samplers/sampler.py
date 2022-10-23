import torch

class Sampler:
    def __init__(self, unconditional_guidance_scale=1):
        self.unconditional_guidance_scale = unconditional_guidance_scale
    
    def set_model(self, model):
        self.model = model
    
    def get_model_output(self, x, t, c=None, unconditional_conditioning=None):
        if not torch.is_tensor(t):
            t = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        if unconditional_conditioning is None or self.unconditional_guidance_scale == 1:
            e_t = self.model.apply_model(x, t, c)
        else:
            e_t_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            e_t = self.model.apply_model(x, t, c)
            e_t = e_t_uncond + self.unconditional_guidance_scale * (e_t - e_t_uncond)
        return e_t
        
    @torch.no_grad()
    def sample(self, shape, cond=None, mask=None, x0=None, x_T=None, unconditional_conditioning=None):
        return x_T
