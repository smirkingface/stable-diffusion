import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import re
import json

from sd.data.utils import load_transforms, apply_transforms

class DirectoryAndFileTagsDataset(Dataset):
    def __init__(self, data_root, mask_key=None, transforms=[], caption_transforms=[]):

        self.data_root    = data_root

        self.image_paths = glob.glob(f'{data_root}/**/*.*', recursive = True)
        self.image_paths = [x for x in self.image_paths if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and (mask_key == None or not x.lower().endswith('.' + mask_key))]

        self.num_images = len(self.image_paths)
        print(f'DirectoryAndFileTagsDataset has {self.num_images} images')
        self._length = self.num_images

        self.mask_key = mask_key

        self.transforms = load_transforms(transforms)
        self.caption_transforms = load_transforms(caption_transforms)

    def __len__(self):
        return self._length

    def load_image(self, i):
        image = Image.open(self.image_paths[i % self.num_images])
        if not image.mode == 'RGB' and not image.mode == 'RGBA':
            image = image.convert('RGB')

        if self.mask_key != None:
            mask_filename = os.path.splitext(self.image_paths[i % self.num_images])[0] + '.' + self.mask_key
            if os.path.exists(mask_filename):
                mask = Image.open(mask_filename).convert('L')
                image.putalpha(mask)

        return image

    def get_caption(self, i):
        filename = os.path.splitext(self.image_paths[i % self.num_images])[0]
        filename = filename[len(f'{self.data_root}/'):].split('.')[0]
        image_path, image_basename = os.path.split(filename)

        # Parse filename/path into tags
        dir_tag = re.sub(r'[\\/]+', ', ', image_path).lower().lstrip().rstrip()

        # NOTE: All of these are removed for tag parsing:
        # * Long hex hashes (at the beginning of the filename)
        # * Solo digits or digits inside of parens
        # * 'left', 'mid', or 'right', as a single isolated piece

        # Example legal path: main_big_tag/some other tag/tag, this tag, thing in background/5e1a57b8b6bac345d9ea6f8b59b468e5-000734-left-other tags, more tags.png
        # Resulting caption: main_big_tag, some other tag, tag, this tag, thing in background, other tags, more tags

        caption = re.sub(r"""(?ix)
            ^[0-9a-f]{20,}- |            # hex hashes
            # solo digits, like 000000.png filenames, or ' (1)', plus screen-split identifiers
            \s* [-(]? \b \d+ \b [-)]? (?:left|mid|right)? -?
        """, '', image_basename).lower().lstrip().rstrip()

        if not caption:
            caption = dir_tag
        else:
            caption = dir_tag + ', ' + caption

        return caption

    def load_info(self, i):
        filename = os.path.splitext(self.image_paths[i % self.num_images])[0] + '.json'
        if os.path.exists(filename):
            return json.load(open(filename, 'r'))
        else:
            return {}

    def __getitem__(self, i):
        image   = self.load_image(i)
        caption = self.get_caption(i)
        info    = self.load_info(i)

        image = apply_transforms(self.transforms, image, info)
        caption = apply_transforms(self.caption_transforms, caption, info)

        example = {'caption': caption}
        if image.mode == 'RGBA':
            image = np.array(image).astype(np.uint8)
            example['image'] = (image[...,:3] / 127.5 - 1.0).astype(np.float32)
            example['mask'] = (image[...,[3]] / 255.0).astype(np.float32)
        else:
            image = np.array(image).astype(np.uint8)
            example['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return example
