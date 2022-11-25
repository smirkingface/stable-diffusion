# SD 2.0 compatibility

This repository has been made compatible with the new SD 2.0 models, both with and without v-parameterization (512-base-ema.ckpt and 768-v-ema.ckpt).

## Training

You can run the new example with:
```
python main.py -c ./configs/finetune_example_sd2.yaml --resume_from_checkpoint ./models/768-v-ema.ckpt
```

If you wish to finetune 512-base-ema.ckpt, add ```parameterization: eps``` to the ```params``` in the ```model``` section of the configuration file.

Note that the 768-v-ema model appears to work way better on generating 768x768 resolution than 512x512 (the default for the ImageLogger, for example). Note that the finetuning example configuration is still set up at to generate 512x512 images.

## Inference

You can run inference with the 768-v-ema model by adding ```--v2``` to the inference command line:
```
python scripts/txt2img.py --v2 --prompt "a smirking face" --ckpt ./models/768-v-ema.ckpt --W 768 --H 768
```
