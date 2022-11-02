import argparse
import torch

from sd.util import traverse_state_dict

scheduler_keys = ['betas', 'alphas_cumprod', 'alphas_cumprod_prev', 'sqrt_alphas_cumprod', 'sqrt_one_minus_alphas_cumprod', 'log_one_minus_alphas_cumprod', 'sqrt_recip_alphas_cumprod', 'sqrt_recipm1_alphas_cumprod', 'posterior_variance', 'posterior_log_variance_clipped', 'posterior_mean_coef1', 'posterior_mean_coef2']

def merge_not_in_source(sd_src, sd_dst, k, state, verbose=True, alpha_new=0.5, merge_vae=False, merge_text_model=False, **kwargs):
    dst_item = sd_dst[k]
    
    if not merge_vae and k.startswith('first_stage_model.'):
        return dst_item
    if not merge_text_model and k.startswith('cond_stage_model.'):
        return dst_item

    # Ignore missing scheduler parameters in merging
    if k in scheduler_keys:
        return dst_item

    if len(dst_item.shape) == 1:
        if verbose:
            print(f'{k}: Bias, merging new weight')
        return alpha_new*dst_item
    else:
        if verbose:
            print(f'{k}: Weight, merging new weight with dirac')
        tmp = alpha_new*dst_item
        torch.nn.init.dirac_(dst_item)
        return (1-alpha_new)*dst_item + tmp

def merge_not_in_dest(sd_src, sd_dst, k, state, verbose=True, **kwargs):
    if k in scheduler_keys:
        if verbose:
            print(f'{k}: Scheduler parameter not in destination, adding')
        return sd_src[k]
    return None

def merge_size_different(sd_src, sd_dst, k, state, verbose=True, alpha=0.5, alpha_new=0.5, merge_vae=False, merge_text_model=False, **kwargs):
    if not merge_vae and k.startswith('first_stage_model.'):
        return sd_dst[k]
    if not merge_text_model and k.startswith('cond_stage_model.'):
        return sd_dst[k]

    src_size = sd_src[k].size()
    dst_size = sd_dst[k].size()
    
    # Assumes weight is convolutional
    # TODO: Support different kernel sizes per dimension
    delta = (dst_size[3] - src_size[3])//2
    pd = torch.nn.ConstantPad2d(delta,0)
    mask = pd(torch.ones_like(sd_src[k]))
    
    if verbose:
        print(f'{k}: Merging')
    return (1-alpha) * pd(sd_src[k]) + mask*alpha*sd_dst[k] + alpha_new*(1-mask)*sd_dst[k]


def merge_matching(sd_src, sd_dst, k, state, verbose=True, alpha=0.5, merge_vae=False, merge_text_model=False, **kwargs):
    if not merge_vae and k.startswith('first_stage_model.'):
        return sd_dst[k]
    if not merge_text_model and k.startswith('cond_stage_model.'):
        return sd_dst[k]

    if verbose:
        print(f'{k}: Merging')
    return (1-alpha) * sd_src[k] + alpha*sd_dst[k]
    
def merge_checkpoint(src_filename, dst_filename, output_filename, alpha=0.5, alpha_new=0.5, merge_vae=False, merge_text_model=False, verbose=True):
    checkpoint_src = torch.load(src_filename, map_location='cpu')
    checkpoint_dst = torch.load(dst_filename, map_location='cpu')
    
    checkpoint_dst['state_dict'],_ = traverse_state_dict(checkpoint_src['state_dict'], checkpoint_dst['state_dict'], func_not_in_source=merge_not_in_source,
                                                         func_matching=merge_matching, func_size_different=merge_size_different,
                                                         alpha=alpha, alpha_new=alpha_new, verbose=verbose,
                                                         merge_vae=merge_vae, merge_text_model=merge_text_model)
    
    remove_keys = ['loops', 'optimizer_states', 'lr_schedulers', 'callbacks']
    for x in remove_keys:
        if x in checkpoint_dst:
            del checkpoint_dst[x]
    
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
        '-a',
        '--alpha',
        type=float,
        default=0.5,
        help='merge alpha',
    )
    parser.add_argument(
        '-an',
        '--alpha_new',
        type=float,
        help='merge alpha (new weights)',
    )
    parser.add_argument(
        '-mv',
        '--merge_vae',
        action='store_true',
        help='merge VAE model (if false, uses the VAE in the dst checkpoint)'
    )
    parser.add_argument(
        '-mt',
        '--merge_text_model',
        action='store_true',
        help='merge text model (if false, uses the text model in the dst checkpoint)'
    )
        
    opt = parser.parse_args()
    
    alpha_new = opt.alpha_new
    if alpha_new == None:
        alpha_new = opt.alpha

    merge_checkpoint(opt.src, opt.dst, opt.output_filename, alpha=opt.alpha, alpha_new=alpha_new, merge_vae=opt.merge_vae, merge_text_model=opt.merge_text_model)
