# Refactored Stable Diffusion
This is a heavily refactored version of the Stable Diffusion development repository (https://github.com/pesser/stable-diffusion/), aiming at providing easier finetuning and modifications of the Stable Diffusion model.

Currently, the repository is almost fully compatible with checkpoints produced with other versions of both the official and development repository of Stable Diffusion. Without modifications to the model parameters, the checkpoints produced by this code are fully backwards-compatible and can be loaded just like the official Stable Diffusion checkpoint (e.g. in the webui).

For a list of changes and notes on compatibility with the original repository, see [changes.md](changes.md).

# Supporting this project
This repository is being developed as part of the SmirkingFace project and you can support its development through Patreon (http://patreon.com/smirkingface). While the SD finetuning for this project aims at NSFW content, the code developed here is general purpose and not biased in any way, nor contains any NSFW content.

You can also support the project by testing, helping to implement new features, and otherwise be involved with the community.

# Getting started with finetuning Stable Diffusion

## Prerequisites
- Finetuning the complete Stable Diffusion model currently requires a GPU with at least 24GB VRAM and at least 32GB system RAM.

## Installation
The easiest way to get started is through Anaconda (https://anaconda.org/).

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
While not the primary focus of this project, improved versions of the text-to-image and image-to-image scripts are provided in [/scripts](/scripts). By default, it will assume you have placed the Stable Diffusion checkpoint in [/models](/models) named `model.ckpt`. The commandline options are mostly the same as the inference scripts provided in the official Stable Diffusion repository.
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
See [docs/examples.md](docs/examples.md)