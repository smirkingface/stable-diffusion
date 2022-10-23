import argparse
import torch

from sd.util import traverse_state_dict

def merge_not_in_source(sd_src, sd_dst, k, state, verbose=True, merge_new=False, alpha_new=0.5, **kwargs):
    dst_item = sd_dst[k]

    if merge_new:
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
    else:
        return dst_item


def merge_size_different(sd_src, sd_dst, k, state, verbose=True, merge_new=False, alpha=0.5, alpha_new=0.5, **kwargs):
    src_size = sd_src[k].size()
    dst_size = sd_dst[k].size()
    
    # Assumes weight is convolutional
    # TODO: Support different kernel sizes per dimension
    delta = (dst_size[3] - src_size[3])//2
    pd = torch.nn.ConstantPad2d(delta,0)
    mask = pd(torch.ones_like(sd_src[k]))
    
    if merge_new:
        if verbose:
            print(f'{k}: Merging (including new weights)')
        return (1-alpha) * pd(sd_src[k]) + mask*alpha*sd_dst[k] + alpha_new*(1-mask)*sd_dst[k]
    else:
        if verbose:
            print(f'{k}: Merging (keeping new weights)')
        return (1-alpha) * pd(sd_src[k]) + mask*alpha*sd_dst[k] + (1-mask)*sd_dst[k]

def merge_matching(sd_src, sd_dst, k, state, verbose=True, alpha=0.5, **kwargs):
    if verbose:
        print(f'{k}: Merging')
    return (1-alpha) * sd_src[k] + alpha*sd_dst[k]
    
def merge_checkpoint(src_filename, dst_filename, output_filename, merge_new=False, alpha=0.5, alpha_new=0.5, verbose=True):
    checkpoint_src = torch.load(src_filename, map_location='cpu')
    checkpoint_dst = torch.load(dst_filename, map_location='cpu')
    
    checkpoint_dst['state_dict'],_ = traverse_state_dict(checkpoint_src['state_dict'], checkpoint_dst['state_dict'], func_not_in_source=merge_not_in_source,
                                                         func_matching=merge_matching, func_size_different=merge_size_different,
                                                         merge_new=merge_new, alpha=alpha, alpha_new=alpha_new, verbose=verbose)
    
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
        '-mn',
        '--merge_new',
        action='store_true',
        help='merge new weights (if False, copies 100%% of new weight)',
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
    
    opt = parser.parse_args()
    
    alpha_new = opt.alpha_new
    if alpha_new == None:
        alpha_new = opt.alpha

    merge_checkpoint(opt.src, opt.dst, opt.output_filename, merge_new=opt.merge_new, alpha=opt.alpha, alpha_new=alpha_new)
