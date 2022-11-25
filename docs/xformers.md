# Using xformers

XFormers (https://github.com/facebookresearch/xformers) are an efficient implementation of Transformer modules, which are used in various places in the Stable Diffusion models. Xformers both reduces VRAM usage and can improve performance as well.

The most current release of xformers is compatible with both full-precision (fp32) and half-precision (fp16) training, although the biggest improvement is expected with half-precision. For example, you can enable Automatic Mixed-Precision during training by specifying ```--precision 16``` on the command line when running main.py.

For the moment, we do require xformers to be explicitly enabled to avoid any compatibility issues (see below).

## Installation

On linux, installing the xformers package can be done through anaconda:
```
conda install xformers -c xformers/label/dev
```
This requires a recent installation of Python (3.9 or 3.10), pytorch (1.12.1 or 1.13) and CUDA (11.3, 11.6, or 11.7). A clean installation using the requirements.yaml in this repository should be compatible.

On Windows, please see https://github.com/facebookresearch/xformers for details on compiling the package from source.

## Enabling xformers

The UNet model now has a parameter ```use_xformers``` to enable using xformers in the UNet attention modules.

For example, the following configuration should be a lot more memory-efficient:
```
    unet_config:
      target: sd.modules.unet.UNetModel
      params:
        use_checkpoint: True
        use_xformers: True
```
