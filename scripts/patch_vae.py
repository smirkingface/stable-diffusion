import os
import torch
import argparse


def patch_vae(filename, vae_filename, output_filename=None):
    size_initial = os.path.getsize(filename)
    checkpoint = torch.load(filename, map_location='cpu')
    checkpoint_vae = torch.load(vae_filename, map_location='cpu')

    sd = checkpoint['state_dict']
    sd_vae = checkpoint_vae['state_dict']
    
    for x in sd_vae:
        k = f'first_stage_model.{x}'
        if 'model_ema' in x:
            continue
        
        assert k in sd and sd[k].shape == sd_vae[x].shape, f'{k} not in sd or wrong shape'
        sd[k] = sd_vae[x]
        
    out_filename = f'{os.path.splitext(filename)[0]}_vae.ckpt'
    out_filename = output_filename or out_filename

    print(f'Saving pruned checkpoint to: {out_filename}')
    torch.save(checkpoint, out_filename)


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
        '-v',
        '--vae',
        type=str,
        required=True,
        help='Path to checkpoint of VAE model',
    )
    parser.add_argument(
        '-o',
        '--output_filename',
        type=str,
        help='Path to save pruned checkpoint',
    )
    
    opt = parser.parse_args()
    patch_vae(opt.filename, opt.vae, opt.output_filename)
