import torch

import importlib
from inspect import isfunction

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f'{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.')
    return total_params

def instantiate_from_config(config):
    return get_obj_from_str(config['target'])(**config.get('params', {}))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, inference=True, swap_ema=False, no_ema=False, verbose=False):
    if verbose:
        print(f'Loading model from {ckpt}')
        
    sd = torch.load(ckpt, map_location='cpu')['state_dict']
    
    if 'params' not in config['model']:
        config['model']['params'] = {}
    
    use_ema = config['model']['params'].get('use_ema', True)

    # TODO: Could check num_updates > 0 to see if EMA weights ever were updated
    
    # Check if the state_dict actually contains EMA weights
    # Exempt num_updates and decay because pruned checkpoints unnecessarily save them
    if use_ema:
        if inference and not any((x.startswith('model_ema.') and x not in ['model_ema.num_updates', 'model_ema.decay']) for x in sd.keys()):
            if not no_ema:
                print('Attempting to load a checkpoint without EMA weights for inference while use_ema is True, setting use_ema to False')
            config['model']['params']['use_ema'] = False
        elif inference and swap_ema:
            # Swap EMA weights into model parameters in state_dict, disable use_ema
            print('Swapping EMA weights in state_dict for EMA-only inference')    
            for k in sd:
                if k.startswith('model.'):
                    key = 'model_ema.' + k[6:].replace('.', '')
                    if key in sd:
                        sd[k] = sd[key]
            
            config['model']['params']['use_ema'] = False
        elif no_ema:
            # Remove EMA weights if requested
            print('Removing EMA weights from state_dict')
            config['model']['params']['use_ema'] = False
        
    # Create model and load weights from checkpoint
    model = instantiate_from_config(config['model'])
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('Missing keys:')
        print(m)
    if len(u) > 0 and verbose:
        print('Unexpected keys:')
        print(u)

    return model

def traverse_state_dict(sd_src, sd_dst, state=None, verbose=True, func_not_in_source=None, func_size_different=None,
                        func_matching=None, func_non_tensor=None, func_not_in_dest=None, **kwargs):
    if state == None:
        state = {}
        
    src_keys = sd_src.keys()
    dst_keys = sd_dst.keys()
        
    for k in dst_keys:
        if not k in sd_src:
            dst_item = sd_dst[k]
            if torch.is_tensor(dst_item):
                dst_size = dst_item.size()
            else:
                dst_size = ''
                
            if verbose:
                print(f'Source is missing {k}, type: {type(sd_dst[k])}, size: {dst_size}')
                
            if func_not_in_source:
                r = func_not_in_source(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                if r != None:
                    sd_dst[k] = r
                else:
                    del sd_dst[k]
        else:
            src_item = sd_src[k]
            dst_item = sd_dst[k]
            
            if torch.is_tensor(src_item) and torch.is_tensor(dst_item):
                src_size = src_item.size()
                dst_size = dst_item.size()
                
                if src_size != dst_size:
                    if verbose:
                        print(f'{k} differs in size: src: {src_size}, dst: {dst_size}')
                        
                    if func_size_different:
                        r = func_size_different(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                        if r != None:
                            sd_dst[k] = r
                        else:
                            del sd_dst[k]
                else:
                    if verbose:
                        print(f'{k} matches')
                        
                    if func_matching:
                        r = func_matching(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                        if r != None:
                            sd_dst[k] = r
                        else:
                            del sd_dst[k]
            else:
                if verbose:
                    print(f'{k} is not a torch Tensor')
                    
                if func_non_tensor:
                    r = func_non_tensor(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                    if r != None:
                        sd_dst[k] = r
                    else:
                        del sd_dst[k]

    for k in src_keys:
        if k not in sd_dst:
            src_item = sd_src[k]
            if torch.is_tensor(src_item):
                src_size = src_item.size()
            else:
                src_size = ''
            
            if verbose:
                print(f'Destination is missing {k}, type: {type(sd_src[k])}, size: {src_size}')
                
            if func_not_in_dest:
                r = func_not_in_dest(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                if r != None:
                    sd_dst[k] = r
    
    return sd_dst, state
