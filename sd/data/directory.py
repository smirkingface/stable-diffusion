import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import json

from sd.data.utils import load_transforms, apply_transforms

class DirectoryDataset(Dataset):
    def __init__(self, data_root, fixed_caption=None, mask_key=None, transforms=[], caption_transforms=[]):

        self.data_root = data_root
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.image_paths = [x for x in self.image_paths if x.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and (mask_key == None or not x.lower().endswith('.' + mask_key))]

        self.num_images = len(self.image_paths)
        print(f'DirectoryDataset has {self.num_images} images')
        self._length = self.num_images 

        self.fixed_caption = fixed_caption
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

    def load_caption(self, i):
        filename = os.path.splitext(self.image_paths[i % self.num_images])[0] + '.txt'
        if self.fixed_caption != None:
            caption = self.fixed_caption
        elif os.path.exists(filename):
            caption = open(filename).read()
        else:
            caption = ''
        return caption

    def load_info(self, i):
        filename = os.path.splitext(self.image_paths[i % self.num_images])[0] + '.json'
        if os.path.exists(filename):
            return json.load(open(filename, 'r'))
        else:
            return {}

    def __getitem__(self, i):
        image = self.load_image(i)
        caption = self.load_caption(i)
        info = self.load_info(i)

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

# Extension to the DirectoryDataset that recursively searches through the data root and uses each subdirectory as a tag to add to the caption
# (either on the front: directory_tag_mode='prepend', or the back: directory_tag_mode='append', or replacing the entire caption: directory_tag_mode='replace')
class TaggedDirectoryDataset(DirectoryDataset):
    def __init__(self, data_root, fixed_caption=None, tag_separator=',', directory_tag_mode='append', mask_key=None, transforms=[], caption_transforms=[]):

        self.data_root = data_root
        self.image_paths = Path(data_root).rglob('*.*')
        self.image_paths = [x for x in self.image_paths if str(x).lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')) and (mask_key == None or not str(x).lower().endswith('.' + mask_key))]
        
        self.directory_tags = [x.relative_to(data_root).parts[:-1] for x in self.image_paths]
        self.tag_separator = tag_separator
        self.directory_tag_mode = directory_tag_mode
        assert(directory_tag_mode in ['append', 'prepend', 'replace'])

        self.num_images = len(self.image_paths)
        print(f'DirectoryDataset has {self.num_images} images')
        self._length = self.num_images 

        self.fixed_caption = fixed_caption
        self.mask_key = mask_key

        self.transforms = load_transforms(transforms)
        self.caption_transforms = load_transforms(caption_transforms)

    def load_caption(self, i):
        filename = os.path.splitext(self.image_paths[i % self.num_images])[0] + '.txt'
        directory_tags = self.directory_tags[i % self.num_images]
        
        if self.directory_tag_mode != 'replace':
            if self.fixed_caption != None:
                caption = self.fixed_caption
            elif os.path.exists(filename):
                caption = open(filename).read()
            else:
                caption = ''
        
        if self.directory_tag_mode == 'append':
            caption = self.tag_separator.join([caption] + list(directory_tags))
        elif self.directory_tag_mode == 'prepend':
            caption = self.tag_separator.join(list(directory_tags) + [caption])
        elif self.directory_tag_mode == 'replace':
            caption = self.tag_separator.join(list(directory_tags))
        return caption