# Examples and advanced use
Described below are some common use-cases and the changes they need in the configuration file.

## Configuration structure and providing custom modules
A common pattern in the configuration file is the presence of dictionaries with `target` and `params` fields (this was inherited from the original SD repository). In modules that accept this structure, this is replaced by a call to the relevant `target`, supplying the `params` as keyword argument.

For example:
Configuration:
```
model:
  target: sd.models.diffusion.StableDiffusion
  params:
    use_checkpoint: True
```
Gets interpreted as:
```
from sd.models.diffusion import StableDiffusion
model = StableDiffusion(use_checkpoint=True)
```

With this pattern you can implement you own modules and provide them as the target, e.g. `target: my_module.MyClass` (for `MyClass` defined in `my_module.py`).

## Repeating a dataset to increase epoch length
1. In the configuration file, wrap your dataset with RepeatedDataset, and provide `<N>` to indicate the number of repeats:
```
target: sd.data.utils.AlternatingDataset
  params:
    dataset:
      target: ...
      params: ...
    repeats: <N>
```

Note: In training deep learning, an epoch is mostly an arbitrary definition that does not impact results much. Training 10 epochs with one dataset or training 1 epoch with one dataset that is repeated 10 times is identical. But the epoch length can influence other things like the frequency of saving checkpoints (but you can also achieve by changing `every_n_epochs` in the `modelcheckpoint` configuration, for example).

## Mixing training data from multiple datasets
1. In the configuration file, wrap your datasets with AlternatingDataset:
```
target: sd.data.utils.AlternatingDataset
  params:
    datasets:
    - target: ...
      params: ...
    - target: ...
      params: ...
```

This will alternately provide images from each dataset. The epoch will end once the smallest dataset has iterated through all its items. Optionally, provide `total_items` in `params` to limit the total number of images generated per epoch.

## Training with non-square images, padded to square images
1. In the configuration file, configure the dataset `transforms` to include ResizePadAndCrop:
```
transforms:
- {target: sd.data.augmentations.ResizePadAndCrop, params: {shape: 512, max_scale_before_crop: 1.5, allow_padding: True, center_if_padded: False}}
```

To test the data loading and augmentation use:
```
python test_dataloader.py -b ./configs/<config>.yaml
```
Prompts and image samples will be logged to `./test_dataloader`.

Parameters:
- `shape`: Final shape of the image after cropping (here, 512x512). Can also be a list of shapes, see next example.
- `max_scale_before_crop`: Maximum scale of the image before cropping, relative to `shape`. (here, 1.5 means images get resized to round(scale\*512), where scale is sampled randomly in [1.0, 1.5], so rescaling randomly in the range [512, 768])
- `allow_padding`: When set to True, will resize non-square images to fit within the desired shape, padding where necessary. When set to False, resize will ensure that the final shape can be cropped out of the resized image without padding.
- `center_if_padded`: When set to True, will use centered crops on padded dimensions, instead of random crops.

## Training with non-square images of variable size
1. In the configuration file, wrap your dataset with VariableShapeDataset:
```
target: sd.data.utils.VariableShapeDataset
  params:
    batch_size: <B>
    total_batches: <N>
    dataset:
      target: ...
```
2. In the configuration file, configure the dataset `transforms` to include ResizePadAndCrop with a list of shapes (or implement your own resizer):
```
transforms:
- {target: sd.data.augmentations.ResizePadAndCrop, params: {shape: [[512, 768], [768, 512]]}}
```

Notes:
- Memory-wise it is most efficient to use shapes with the same number of pixels, but this will work even with variable numbers of pixels.
- By default, ResizePadAndCrop will match the aspect ratio of the image from the data to one of the provided shapes.
- VariableShapeDataset will collate images with the same shape in buffers and yield the buffer once it reaches the desired batch size (`<B>`). You will also need to provide the total number of batches per epoch provided this way (`<N>`). Using many different shapes combined with a high `<B>` and low `<N>` will result in low efficiency.

## Using the 8-bit Adam optimizer
Note: Currently linux-only

1. Install bitsandbytes: ```pip install bitsandbytes```
2. In the configuration file, configure `optimizer_config` to use Adam8bit:
```
optimizer_config:
  target: bitsandbytes.optim.Adam8bit
```
