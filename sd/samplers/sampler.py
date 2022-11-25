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
            model_output = self.model.apply_model(x, t, c)
        else:
            model_output_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = self.model.apply_model(x, t, c)
            model_output = model_output_uncond + self.unconditional_guidance_scale * (model_output - model_output_uncond)
        
        if self.model.parameterization == 'v':
            e_t = self.model.schedule.predict_eps_from_z_and_v(x, t-1, model_output)
            pred_x0 = self.model.schedule.predict_start_from_z_and_v(x, t-1, model_output)
        else:
            e_t = model_output
            pred_x0 = self.model.schedule.predict_start_from_eps(x, t-1, e_t)

        return e_t, pred_x0
        
    @torch.no_grad()
    def sample(self, shape, cond=None, mask=None, x0=None, x_T=None, unconditional_conditioning=None):
        return x_T
