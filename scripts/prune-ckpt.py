import os
import torch
import argparse


def prune_checkpoint(filename, output_filename=None, only_ema=False, remove_ema=False, fp16=False):
    print(f'Pruning checkpoint: {filename}')
    size_initial = os.path.getsize(filename)
    checkpoint = torch.load(filename, map_location='cpu')

    remove_keys = ['loops', 'optimizer_states', 'lr_schedulers', 'callbacks']
    for k in remove_keys:
        if k in checkpoint:
            del checkpoint[k]

    if 'global_step' in checkpoint:
        print(f"This is global step {checkpoint['global_step']}.")
    
    sd = checkpoint['state_dict']
    if only_ema:
        # Overwrite EMA weights into model parameters
        keys = list(sd.keys())
        ema_keys = set(['model_ema.num_updates', 'model_ema.decay'])
        for k in keys:
            if k.startswith('model.'):
                key = 'model_ema.' + k[6:].replace('.', '')
                if key in sd:
                    ema_keys.add(key)
                    sd[k] = sd[key]
        
        # Delete EMA keys, check if any were unused
        for k in keys:
            if k.startswith('model_ema.'):
                if k not in ema_keys:
                    print(f'  {k} found in model_ema, but no matching parameter in model')
                del sd[k]
        
        out_filename = f'{os.path.splitext(filename)[0]}_ema.ckpt'
    elif remove_ema:
        # Delete EMA keys
        keys = list(sd.keys())
        for k in keys:
            if k.startswith('model_ema.'):
                del sd[k]
        
        out_filename = f'{os.path.splitext(filename)[0]}_noema.ckpt'
    else:
        out_filename = f'{os.path.splitext(filename)[0]}_pruned.ckpt'
    
    # Convert weights to FP16 if requested
    if fp16:
        exclude = ['model_ema.num_updates', 'model_ema.decay', 'model.betas', 'model.alphas_cumprod',
                   'model.alphas_cumprod_prev', 'model.sqrt_alphas_cumprod', 'model.sqrt_one_minus_alphas_cumprod',
                   'model.log_one_minus_alphas_cumprod', 'model.sqrt_recip_alphas_cumprod', 'model.sqrt_recipm1_alphas_cumprod',
                   'model.posterior_variance', 'model.posterior_log_variance_clipped', 'model.posterior_mean_coef1',
                   'model.posterior_mean_coef2']

        for k in sd:
            if torch.is_tensor(sd[k]) and k not in exclude:
                if sd[k].dtype == torch.float32:
                    sd[k] = sd[k].to(torch.float16)
        
        out_filename = f'{os.path.splitext(filename)[0]}-fp16.ckpt'

    out_filename = output_filename or out_filename

    print(f'Saving pruned checkpoint to: {out_filename}')
    torch.save(checkpoint, out_filename)
    newsize = os.path.getsize(out_filename)
    msg = f'New ckpt size: {newsize*1e-9:.2f} GB. ' + \
          f'Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states'
    if only_ema:
        msg += ' and non-EMA weights'
    elif remove_ema:
        msg += ' and EMA weights'
    if fp16:
        msg += ' and converting weights to fp16'
    print(msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--filename',
        type=str,
        required=True,
        help='Path to checkpoint of model',
    )
    parser.add_argument(
        '-o',
        '--output_filename',
        type=str,
        help='Path to save pruned checkpoint',
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='Store weights in 16-bit precision',
    )
    parser.add_argument(
        '--only_ema',
        action='store_true',
        help='Store only EMA weights',
    )
    parser.add_argument(
        '--remove_ema',
        action='store_true',
        help='Remove EMA weights',
    )
    
    opt = parser.parse_args()
    prune_checkpoint(opt.filename, opt.output_filename, only_ema=opt.only_ema, remove_ema=opt.remove_ema, fp16=opt.fp16)
