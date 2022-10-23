# Refactored Stable Diffusion
This is a heavily refactored version of the Stable Diffusion development repository (https://github.com/pesser/stable-diffusion/), aiming at providing easier finetuning and modifications of the Stable Diffusion model.

Currently, the repository is almost fully compatible with checkpoints produced with other versions of both the official and development repository of Stable Diffusion. Without modifications to the model parameters, the checkpoints produced by this code are fully backwards-compatible and can be loaded just like the official Stable Diffusion checkpoint (e.g. in the webui).

The largest changes include:
General:
- Removed most unused code and reduced the package dependencies
- Support for latest Pytorch-Lightning (tested with 1.7.7) and Pytorch (tested with 1.11.0) releases and removed most deprecation warnings
- Much shorter configuration files by settting the default parameters for all modules to match Stable Diffusion.

Training:
- Quality of life features for resuming training from both the Stable Diffusion checkpoint (without optimizer states, with or without EMA weights) and regular checkpoints.
- More control over how to resume from a checkpoint: Optionally reset the optimizer (`--reset_optimizer`), learning rate (`--reset-lr`), and/or callback states (`--reset_callbacks`). Resuming from a Stable Diffusion checkpoint will automatically set these flags to get the expected result.
- Added support for loading training data from a directory containing image files (.jpg, .jpeg, .png) and .txt files containing captions, and an optional training mask image (e.g. to exclude padded regions of images and/or to exclude watermarks).
- Added a unified data augmentation parameters to all supplied data loaders, supporting augmentation of both images and image captions. These can be for example torchvision modules, some modules provided in this repository (e.g. RescalePadAndCrop, CaptionShuffle), or your own implementations. In your own implementations is also possible to use additional image information, for example from a .json file, which can support turning augmentations on/off on a single image basis, or generating random captions, for example.
- Added support for masked loss calculation, to support training with non-square images.

Sampling:
- Separated DDPM scheduling and sampling from the PL training module, and added more samplers that can also be easily used to track finetuning progress during training.
- Fixed a bug in the DDIM and PLMS samplers. DDIM with eta=1.0 is now almost identical to the "Euler Ancestral" method from k-diffusion most are familiar with.

Model:
- A modified UNet model with support for changing to larger convolution kernels, and scripts for preparing the SD checkpoint for this change.

Planned improvements:
- Integrating a fused/segmented implementation of the attention modules to improve performance and reduce memory requirements
- Support for freezing arbitrary layers
- Integrating textual inversion and dreambooth-style training
- Finetuning of the autoencoder

Compatibility notes:
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

# Supporting this project
This repository is being developed as part of the SmirkingFace project and you can support its development through Patreon ([http://patreon.com/smirkingface]). While the SD finetuning for this project aims at NSFW content, the code developed here is general purpose and not biased in any way, nor contains any NSFW content.

You can also support the project by testing, helping to implement new features, and otherwise be involved with the community.

# Getting started with finetuning Stable Diffusion

## Prerequisites
- Finetuning the complete Stable Diffusion model currently requires a GPU with 24GB VRAM or more and 32GB system RAM.

## Installation
The easiest way to get started is through Anaconda [https://anaconda.org/].

Create and activate an Anaconda environment with:
```
conda env create -f environment.yaml
conda activate sd
```

You can also install the required packages yourself (see requirements.txt, there are not that many!), just be mindful that a more recent version of pytorch-lightning is required to run this code, compared to the official SD code.

## Training
To test whether you are set to finetune Stable Diffusion, you can run a test with:
```
python main.py -c ./configs/dev.yaml --resume_from_checkpoint ./models/model.ckpt
```
(where model.ckpt is the SD checkpoint). While this test does not produce any useable results, it will still encounter errors if you are not set up correctly or if your GPU hardware is not sufficient (24GB VRAM required). You can put use_ema to True in the configuration to test whether you can train while keeping EMA (Exponential Moving Average) weights.

If this runs, you can set up your own configuration, for example following `finetune_example.yaml`, and run it:
```
python main.py -c ./configs/finetune_example.yaml --resume_from_checkpoint ./models/model.ckpt
```

Training can be interrupted by pressing CTRL+C once (!), which will save a final checkpoint. Please be patient for the checkpoint to save, this can take a while. Note that although interrupting will save a checkpoint, it is highly recommended to keep regular checkpoints using the `modelcheckpoint` configuration under `lightning`.

You can resume an interrupted training run with:
```
python main.py -c ./configs/<config>.yaml -r ./logs/[directory_name]
```
Optionally, you can supply `--reset_lr` to reset the learning rate and learning rate scheduler, which are then initialized from the configuration, instead of from the checkpoint. This allows changing learning rate mid-experiment.

Pytorch-lightning supports many additional command-line arguments for training on varied kinds of hardware and with different training strategies: See ```python main.py --help``` and https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html. These options can also be supplied in the configuration file under `trainer` in  `lightning`.


# Running text-to-image and image-to-image
While not the primary focus of this project, improved versions of the text-to-image and image-to-image scripts are provided in [/scripts]. By default, it will assume you have placed the Stable Diffusion checkpoint in [/models] named `model.ckpt`. The commandline options are mostly the same as the inference scripts provided in the official Stable Diffusion repository.
By default these scripts runs the fixed DDIM sampler with eta=1.0 and 50 steps, which performs very comparably to the Euler Ancestral sampler from k-diffusion.
Do note that img2img.py always runs the requested number of inference steps (`--steps`), whereas the original made this dependent on the `--strength` parameter (lower strength implied fewer steps).

Run with:
```
python scripts/txt2img.py --prompt "a smirking face"
```

Running the SmirkingFace models:
```
python scripts/txt2img.py --prompt "a smirking face" --ckpt ./models/SF_EB_1.0.ckpt
python scripts/txt2img.py --prompt "a smirking face" --ckpt ./models/SF_EBL_1.0.ckpt --config ./configs/large_inference.yaml
```

## Examples and advanced use
See [docs/examples]