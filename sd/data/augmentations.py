from PIL import Image
import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as F

import random


# Shuffle a comma-separate text caption. Keep max_len items (None = no limit), and shuffle items. If jitter is an integer, shuffle
# positions by at most jitter positions, useful to keep an approximate ordering.
class ShuffleCaption:
    def __init__(self, jitter=None, max_len=None):
        self.jitter = jitter
        self.max_len = max_len
        
    def __call__(self, caption):
        parts = caption.split(',')
        parts = [x.strip() for x in parts]
        
        # Random subsample which maintains order
        if self.max_len != None and len(parts) > self.max_len:
            parts = [x[1] for x in sorted(random.sample(list(enumerate(parts)), self.max_len))]
        
        if self.jitter == None:
            random.shuffle(parts)
        else:
            # Jitter positions
            parts = [x[1] for x in sorted([(i + random.randint(-self.jitter, self.jitter),x) for i,x in enumerate(parts)], key=lambda x:x[0])]
        return ', '.join(parts)


# Replace synonyms within a caption.
# Define synonyms as either a dictionary or a filename to yaml file containing the dictionary. The key of the dictionary
# is a word, the value can be either a string (a single synonym) or a list of strings (multiple synonyms). Keep in mind
# that the strings are case-sensitive.
# For example (in yaml format):
# house: home
# car: [car, automobile, wagon]
# 'a car': car
# 
# If unify==False, ReplaceSynonyms will replace the key word with a random synonym. If unify==True, all synonyms will be
# replaced by the key word. If match_tag==True, only comma-separated (or tag_separator separated) tags will be matched, which
# can be multiple words. If match_tag==False, synonyms will be matched as complete words (but synonyms can contain spaces and otherwise
# characters).
# Note that if you want to replace a randomly between all occurences of a synonym list, each synonym must be listed as a
# key word, for example:
# car: [car, automobile, wagon]
# automobile: [car, automobile, wagon] 
# wagon: [car, automobile, wagon]
class ReplaceSynonyms:
    def __init__(self, synonyms, match_tag=False, tag_separator=',', unify=False):
        self.unify = unify
        if isinstance(synonyms, str):
            synonyms = yaml.safe_load(open(synonyms, 'r'))
        assert isinstance(synonyms, dict), 'ReplaceSynonyms: synonyms must be a dictionary object'

        if unify:
            # "Invert" synonyms
            unify_rep = {}
            for word,synonyms in synonyms.items():
                if isinstance(synonyms, list):
                    for synonym in synonyms:
                        if synonym not in unify_rep:
                            unify_rep[synonym] = word
                else:
                    if synonyms not in unify_rep:
                        unify_rep[synonyms] = word
            synonyms = unify_rep
        
        self.replacer = self.make_replacer(synonyms, match_tag, tag_separator)

    # Make a function that performs string replacements
    def make_replacer(self, replacements, match_tag, tag_separator=','):
        tag_separator = re.escape(tag_separator)
        def replace_match(m):
            r = replacements[m.group(2)]
            if isinstance(r, list):
                r = random.choice(r)
            return m.group(1) + r
        
        # Non-matching regex for each replacement that needs to be matched
        re_rep = ['(?:' + re.escape(k) + ')' for k in replacements]

        if match_tag:
            # ((?:(?:^|,))\s*) = Preceded by start of string or separator and optional whitespace
            # (?=(?:\s*(?:,|$))) = Lookahead (not matched), has to be followed by optional whitespace and end of string or separator
            pattern = re.compile('((?:(?:^|'+tag_separator+'))\s*)(' + ('|'.join(re_rep)) + ')(?=(?:\s*(?:'+tag_separator+'|$)))')
        else:
            # \b = word boundary
            pattern = re.compile('(\\b)(' + ('|'.join(re_rep)) + ')(?=\\b)')
        return lambda caption: pattern.sub(replace_match, caption)
        
    def __call__(self, caption):
        return self.replacer(caption)

# Augmentation module that resizes, pads, and crops images (whichever apply)
# Final image will be of shape 'shape', which can be:
#   A single integer (square image, e.g. 512 -> 512x512)
#   A list (non-square image, e.g. [512, 768])
#   A list of lists (multiple shapes, e.g. [[512,512], [512,768]])
# Resize: Resizes to desired shape, with a random scale between 1.0 and max_scale_before_crop (i.e. always larger than final shape)
# Pad: If allow_padding, resizes uses the largest dimension for determining the scale, otherwise it uses the smallest dimension
# Crop: Randomly crops the final shape out of the resized image. If center_if_padded, the padded dimension be center-cropped
# If allow_padding, the resulting image will contain a valid mask in the alpha channel
# ignore_alpha: Ignores any alpha channel already present in the loaded image
class ResizePadAndCrop:
    def __init__(self, shape=512, max_scale_before_crop=1.0, allow_padding=False, center_if_padded=True,
                 match_aspect_ratio=True, match_closest_aspect_ratio=False, closest_aspect_ratio_eps=1e-6, ignore_alpha=False):
        self.shape = shape
        if type(self.shape) == int:
            self.shape = [self.shape, self.shape]
        if type(self.shape[0]) != list:
            self.shape = [self.shape]
            
        self.max_scale_before_crop = max_scale_before_crop
        self.allow_padding = allow_padding
        
        self.shapes_w = []
        self.shapes_h = []
        for x in self.shape:
            if x[0] >= x[1]:
                self.shapes_w.append(x)
            if x[0] <= x[1]:
                self.shapes_h.append(x)
        
        self.match_aspect_ratio = match_aspect_ratio
        self.match_closest_aspect_ratio = match_closest_aspect_ratio
        self.closest_aspect_ratio_eps = closest_aspect_ratio_eps
        if match_closest_aspect_ratio:
            self.shape_ars = np.array([x[0] / x[1] for x in self.shape])
            
        self.ignore_alpha = ignore_alpha
        
        self.resize = transforms.Resize(size=1, interpolation=transforms.InterpolationMode.BICUBIC)
        if allow_padding:
            self.crop = RandomCropMasked(size=1, center_if_padded=center_if_padded, ignore_alpha=ignore_alpha)
        else:
            self.crop = transforms.RandomCrop(size=1)

    def __call__(self, image):
        # Throw away alpha channel if requested
        if self.ignore_alpha and image.mode == 'RGBA':
            image = image.convert('RGB')
            
        # Pick random shape, matching aspect ratio if requested
        if self.match_aspect_ratio:
            if self.match_closest_aspect_ratio:
                ar_image = image.size[0] / image.size[1]
                # Find shape with closest aspect ratio (TODO: ignores whether it pads or crops, maybe include?)
                ratio = np.stack((ar_image / self.shape_ars,self.shape_ars / ar_image)).max(axis=0)
                inds = np.where(ratio < ratio.min() + self.closest_aspect_ratio_eps)[0]

                shape = random.choice([self.shape[x] for x in inds])
            else:
                if not self.shapes_w == [] and (image.size[0] > image.size[1] or self.shapes_h == []):
                    shape = random.choice(self.shapes_w)
                else:
                    shape = random.choice(self.shapes_h)
        else:
            shape = random.choice(self.shape)


        # Calculate crop and resize shapes
        # Note: torchvision transforms take (h,w) shapes
        self.crop.size = (shape[1],shape[0])
        ar_shape = shape[0] / shape[1]
        ar_image = image.size[0] / image.size[1]
        
        f = self.max_scale_before_crop
        if self.allow_padding:
            if ar_shape > ar_image:
                h = random.randint(shape[1], min(round(shape[1]*f), image.size[1]))
                w = round(image.size[0]/image.size[1] * h)
            else:
                w = random.randint(shape[0], min(round(shape[0]*f), image.size[0]))
                h = round(image.size[1]/image.size[0] * w)
        else:
            if ar_shape < ar_image:
                h = random.randint(shape[1], min(round(shape[1]*f), image.size[1]))
                w = round(image.size[0]/image.size[1] * h)
            else:
                w = random.randint(shape[0], min(round(shape[0]*f), image.size[0]))
                h = round(image.size[1]/image.size[0] * w)
        self.resize.size = (h,w)
        
        # Apply resize and crop
        if image.mode == 'RGBA':
            mask = image.split()[3]
            mask = self.resize(mask)
            image = image.convert('RGB')
            image = self.resize(image)
            image.putalpha(mask)
        else:
            image = self.resize(image)
        image = self.crop(image)
        
        return image
    
# Random crop augmentation, also provides a mask in the alpha channel of the augmented image
# Applies a different padding mode to the image (padding_mode) and the mask (always zero-padding)
class RandomCropMasked(transforms.RandomCrop):
    def __init__(self, size, padding=None, center_if_padded=False, padding_mode='reflect', ignore_alpha=False):
        super().__init__(size, padding=padding, pad_if_needed=True, padding_mode=padding_mode)
        self.center_if_padded = center_if_padded
        self.ignore_alpha = ignore_alpha
        
    def forward(self, img):
        tmp = img.split()
        if not self.ignore_alpha and len(tmp) == 4:
            mask = tmp[3]
        else:
            mask = Image.new('L', img.size, 255)
        img = img.convert('RGB')

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)
            mask = F.pad(mask, self.padding, 0, 'constant')

        # Newer torchvision uses get_dimensions instead of get_image_size
        if hasattr(F, 'get_dimensions'):
            _, height, width = F.get_dimensions(img)
        else:
            width, height = F.get_image_size(img)
        
        w_padded = False
        h_padded = False
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, 0, 'constant')
            w_padded = True
            
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)
            mask = F.pad(mask, padding, 0, 'constant')
            h_padded = True

        i, j, h, w = self.get_params(img, self.size)
        if self.center_if_padded:
            if w_padded:
                j = (self.size[1] - width) // 2
            if h_padded:
                i = (self.size[0] - height) // 2
        
        r = F.crop(img, i, j, h, w)
        # r.convert('RGB').save(f'test_{random.randint(0,100)}.png')
        r.putalpha(F.crop(mask, i, j, h, w))
        return r
