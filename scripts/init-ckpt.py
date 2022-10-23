import argparse
import torch

from sd.util import traverse_state_dict

def init_not_in_source(sd_src, sd_dst, k, state, verbose=True, sigma=0, **kwargs):
    dst_item = sd_dst[k]
    
    # TODO: Support for non-convolutional added parameters
    if len(dst_item.shape) == 1:
        # Assumes k is bias
        if verbose:
            print(f'{k}: Assigning noised bias')
        return sigma*torch.randn_like(dst_item)
    else:
        # Assumes k is convolutional
        if verbose:
            print(f'{k}: Assigning noised dirac')
        torch.nn.init.dirac_(sd_dst[k])
        return sd_dst[k] + sigma*torch.randn_like(dst_item)

def init_size_different(sd_src, sd_dst, k, state, verbose=True, sigma=0, **kwargs):
    src_size = sd_src[k].size()
    dst_size = sd_dst[k].size()
    
    # Assumes weight is convolutional
    # TODO: Support different kernel sizes per dimension
    delta = (dst_size[3] - src_size[3])//2
    pd = torch.nn.ConstantPad2d(delta,0)
    mask = pd(torch.ones_like(sd_src[k]))
    
    if verbose:
        print(f'{k}: Extending source and filling in with noise')
    return pd(sd_src[k]) + (1-mask)*sigma*torch.randn_like(sd_dst[k])

def init_matching(sd_src, sd_dst, k, state, verbose=True, sigma=0, **kwargs):
    return sd_src[k]

def initialize_checkpoint(src_filename, dst_filename, output_filename, sigma=0, verbose=True):
    # TODO: Allow initializing a new model from config (rather than a checkpoint of a new model)
    checkpoint_src = torch.load(src_filename, map_location='cpu')
    checkpoint_dst = torch.load(dst_filename, map_location='cpu')
    
    checkpoint_dst['state_dict'],_ = traverse_state_dict(checkpoint_src['state_dict'], checkpoint_dst['state_dict'], func_not_in_source=init_not_in_source,
                                                         func_size_different=init_size_different, func_matching=init_matching, sigma=sigma, verbose=verbose)
    
    remove_keys = ['loops', 'optimizer_states', 'lr_schedulers', 'callbacks']
    for x in remove_keys:
        if x in checkpoint_dst:
            del checkpoint_dst[x]
    
    if 'epoch' in checkpoint_src:
        checkpoint_dst['epoch'] = checkpoint_src['epoch']
    if 'global_step' in checkpoint_src:
        checkpoint_dst['global_step'] = checkpoint_src['global_step']

    torch.save(checkpoint_dst, output_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s',
        '--src',
        type=str,
        required=True,
        help='Filename of the checkpoint to copy from',
     )
    parser.add_argument(
        '-d',
        '--dst',
        type=str,
        required=True,
        help='Filename of the checkpoint to copy to',
    )
    parser.add_argument(
        '-o',
        '--output_filename',
        type=str,
        required=True,
        help='Output filename',
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0.0,
        help='Noise sigma to use for initialization of new weights',
    )
    
    opt = parser.parse_args()
    initialize_checkpoint(opt.src, opt.dst, opt.output_filename, sigma=opt.sigma)
