# List of changes

## General
- Removed most unused code and reduced the package dependencies
- Support for latest Pytorch-Lightning (tested with 1.7.7) and Pytorch (tested with 1.11.0) releases and removed most deprecation warnings
- Much shorter configuration files by settting the default parameters for all modules to match Stable Diffusion.

## Training
- Quality of life features for resuming training from both the Stable Diffusion checkpoint (without optimizer states, with or without EMA weights) and regular checkpoints.
- More control over how to resume from a checkpoint: Optionally reset the optimizer (`--reset_optimizer`), learning rate (`--reset-lr`), and/or callback states (`--reset_callbacks`). Resuming from a Stable Diffusion checkpoint will automatically set these flags to get the expected result.
- Added support for loading training data from a directory containing image files (.jpg, .jpeg, .png) and .txt files containing captions, and an optional training mask image (e.g. to exclude padded regions of images and/or to exclude watermarks).
- Added a unified data augmentation parameters to all supplied data loaders, supporting augmentation of both images and image captions. These can be for example torchvision modules, some modules provided in this repository (e.g. RescalePadAndCrop, CaptionShuffle), or your own implementations. In your own implementations is also possible to use additional image information, for example from a .json file, which can support turning augmentations on/off on a single image basis, or generating random captions, for example.
- Added support for masked loss calculation, to support training with non-square images.

## Sampling
- Separated DDPM scheduling and sampling from the PL training module, and added more samplers that can also be easily used to track finetuning progress during training.
- Fixed a bug in the DDIM and PLMS samplers. DDIM with eta=1.0 is now almost identical to the "Euler Ancestral" method from k-diffusion most are familiar with.

## Model
- A modified UNet model with support for changing to larger convolution kernels, and scripts for preparing the SD checkpoint for this change.

# Compatibility notes
- A newer version of Pytorch-Lightning is strictly necessary to run this code.
- Configuration files for both training and inference are not compatible with the original SD repository.
- Configuration entries can not be partially updated from the defaults. E.g. setting a new optimizer or model checkpoint configuration requires you to provide all parameters. This may change in the future.
- The commandline option for the configuration file (`-b` or `--base`) has been removed and replaced with `-c` or `--config`.
- The commandline option for enabling training (`-t` or `--train`) has been removed, training is always performed.
- The commandline option `--resume_from_checkpoint` is now fully compatible with resuming from the original SD checkpoint, no more need for the `--actual_resume` trick from the Textual Inversion repository.
- The command line option `--scale_lr` has been removed and replaced with `--disable_scale_lr`. Learning rate scaling is and was on by default, now the commandline option is just there to turn it off.
- The commandline option `--gpus` changed to `--devices` in the more recent Pytorch-Lightning release.
- By default, this repository uses a modified ModelCheckpoint function that only saves last.ckpt every N epochs if `every_n_epochs` is set, instead of saving every epoch. This is particularly convenient for smaller datasets (i.e. with short epochs).
- This repository changed the default Pytorch-Lightning logger to CSV (modify `logger` in the `lightning` configuration to change back to other loggers).

# Planned improvements
- Integrating a fused/segmented implementation of the attention modules to improve performance and reduce memory requirements
- Support for freezing arbitrary layers
- Integrating textual inversion and dreambooth-style training
- Finetuning of the autoencoder