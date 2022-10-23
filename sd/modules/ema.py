import torch
from torch import nn

# TODO: If checkpoint has no EMA weights, these will be initialized as noise, instead of the actual weights from the model
# How to fix?
# TODO: CPU EMA (keep EMA weights on CPU, keep in mind that devices may not match when swapping/updating)
# TODO: Function to set parameters after model load_state_dict without EMA, or with --reset-ema
class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().data)
                
    def reset(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                shadow_params[self.m_name2s_name[key]].data.copy_(m_param[key].data)
            else:
                assert not key in self.m_name2s_name
        
        self.num_updates.zero_()
    
    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            # TODO: This warmup should depend on decay?
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert key not in self.m_name2s_name

    def swap(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                x = m_param[key]
                y = shadow_params[self.m_name2s_name[key]]
                x.data, y.data = y.data, x.data
            else:
                assert key not in self.m_name2s_name