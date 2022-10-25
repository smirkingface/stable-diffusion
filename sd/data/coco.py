import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from abc import abstractmethod

from sd.data.utils import load_transforms, apply_transforms

class CocoBase(Dataset):
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, dataroot="", datajson="", given_files=None, transforms=[], caption_transforms=[]):
        self.split = self.get_split()

        data_json = datajson
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()

        assert data_json.split("/")[-1] in [f"captions_train{self.year()}.json",
                                            f"captions_val{self.year()}.json"]

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")

            if given_files is not None:
                if pngfilename in given_files:
                    self.labels["image_ids"].append(imgdir["id"])
            else:
                self.labels["image_ids"].append(imgdir["id"])
        print(f'Coco has {len(self.labels["image_ids"])} images')

        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            self.img_id_to_captions[capdir["image_id"]].append(capdir["caption"])

        self.transforms = load_transforms(transforms)
        self.caption_transforms = load_transforms(caption_transforms)

    @abstractmethod
    def year(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.labels["image_ids"])

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        image = Image.open(img_path)
        if not image.mode == 'RGB' and not image.mode == 'RGBA':
            image = image.convert('RGB')
        
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        
        image = apply_transforms(self.transforms, image)
        caption = apply_transforms(self.caption_transforms, caption)
        
        example = {'caption': caption}
        if image.mode == 'RGBA':
            image = np.array(image).astype(np.uint8)
            example['image'] = (image[...,:3] / 127.5 - 1.0).astype(np.float32)
            example['mask'] = (image[...,[3]] / 255.0).astype(np.float32)
        else:
            image = np.array(image).astype(np.uint8)
            example['image'] = (image / 127.5 - 1.0).astype(np.float32)
        return example
        
class CocoImagesAndCaptionsTrain2017(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="../data/coco/train2017",
                         datajson="../data/coco/annotations/captions_train2017.json",
                         **kwargs)

    def get_split(self):
        return "train"

    def year(self):
        return '2017'

class CocoImagesAndCaptionsValidation2017(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="../data/coco/val2017",
                         datajson="../data/coco/annotations/captions_val2017.json",
                         **kwargs)

    def get_split(self):
        return "validation"

    def year(self):
        return '2017'

class CocoImagesAndCaptionsTrain2014(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="data/coco/train2014",
                         datajson="data/coco/annotations2014/annotations/captions_train2014.json",
                         **kwargs)

    def get_split(self):
        return "train"

    def year(self):
        return '2014'

class CocoImagesAndCaptionsValidation2014(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, **kwargs):
        super().__init__(dataroot="data/coco/val2014",
                         datajson="data/coco/annotations2014/annotations/captions_val2014.json",
                         **kwargs)

    def get_split(self):
        return "validation"

    def year(self):
        return '2014'
