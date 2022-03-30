import cv2
import inspect
import numpy as np
from PIL import Image, ImageFilter


import torch
from torchvision import transforms as _transforms

from mmdet3d.utils import build_from_cfg
from mmdet.datasets import PIPELINES

import mmcv
import random
import math


@PIPELINES.register_module()
class ColorJitter(object):
    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2):
        self.trans = _transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)

    def __call__(self, input_dict):
        img = input_dict["img"]
        img = Image.fromarray(img)
        img = self.trans(img)
        img = np.array(img)
        input_dict["img"] = img
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomGrayscale(object):
    def __init__(self, p=0.2):
        self.trans = _transforms.RandomGrayscale(p=p)

    def __call__(self, input_dict):
        img = input_dict["img"]
        img = Image.fromarray(img)
        img = self.trans(img)
        img = np.array(img)
        input_dict["img"] = img
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str

@PIPELINES.register_module()
class RandomResizedCrop(object):
    """Crop the given image to random size and aspect ratio.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a
    random aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio
    is made. This crop is finally resized to given size.
    Args:
        size (sequence | int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        scale (tuple): Range of the random size of the cropped image compared
            to the original image. Defaults to (0.08, 1.0).
        ratio (tuple): Range of the random aspect ratio of the cropped image
            compared to the original image. Defaults to (3. / 4., 4. / 3.).
        max_attempts (int): Maxinum number of attempts before falling back to
            Central Crop. Defaults to 10.
        efficientnet_style (bool): Whether to use efficientnet style Random
            ResizedCrop. Defaults to False.
        min_covered (Number): Minimum ratio of the cropped area to the original
             area. Only valid if efficientnet_style is true. Defaults to 0.1.
        crop_padding (int): The crop padding parameter in efficientnet style
            center crop. Only valid if efficientnet_style is true.
            Defaults to 32.
        interpolation (str): Interpolation method, accepted values are
            'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'. Defaults to
            'bilinear'.
        backend (str): The image resize backend type, accpeted values are
            `cv2` and `pillow`. Defaults to `cv2`.
    """

    def __init__(self,
                 size,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4., 4. / 3.),
                 max_attempts=10,
                 efficientnet_style=False,
                 min_covered=0.1,
                 crop_padding=32,
                 interpolation='bilinear',
                 backend='cv2'):
        if efficientnet_style:
            assert isinstance(size, int)
            self.size = (size, size)
            assert crop_padding >= 0
        else:
            if isinstance(size, (tuple, list)):
                self.size = size
            else:
                self.size = (size, size)
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max). '
                             f'But received scale {scale} and rato {ratio}.')
        assert min_covered >= 0, 'min_covered should be no less than 0.'
        assert isinstance(max_attempts, int) and max_attempts >= 0, \
            'max_attempts mush be of typle int and no less than 0.'
        assert interpolation in ('nearest', 'bilinear', 'bicubic', 'area',
                                 'lanczos')
        if backend not in ['cv2', 'pillow']:
            raise ValueError(f'backend: {backend} is not supported for resize.'
                             'Supported backends are "cv2", "pillow"')

        self.scale = scale
        self.ratio = ratio
        self.max_attempts = max_attempts
        self.efficientnet_style = efficientnet_style
        self.min_covered = min_covered
        self.crop_padding = crop_padding
        self.interpolation = interpolation
        self.backend = backend

    @staticmethod
    def get_params(img, scale, ratio, max_attempts=10):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (ndarray): Image to be cropped.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maxinum number of attempts before falling back
                to central crop. Defaults to 10.
        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height = img.shape[0]
        width = img.shape[1]
        area = height * width

        for _ in range(max_attempts):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            target_width = int(round(math.sqrt(target_area * aspect_ratio)))
            target_height = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < target_width <= width and 0 < target_height <= height:
                ymin = random.randint(0, height - target_height)
                xmin = random.randint(0, width - target_width)
                ymax = ymin + target_height - 1
                xmax = xmin + target_width - 1
                return ymin, xmin, ymax, xmax

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            target_width = width
            target_height = int(round(target_width / min(ratio)))
        elif in_ratio > max(ratio):
            target_height = height
            target_width = int(round(target_height * max(ratio)))
        else:  # whole image
            target_width = width
            target_height = height
        ymin = (height - target_height) // 2
        xmin = (width - target_width) // 2
        ymax = ymin + target_height - 1
        xmax = xmin + target_width - 1
        return ymin, xmin, ymax, xmax

    # https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/data.py # noqa
    @staticmethod
    def get_params_efficientnet_style(img,
                                      size,
                                      scale,
                                      ratio,
                                      max_attempts=10,
                                      min_covered=0.1,
                                      crop_padding=32):
        """Get parameters for ``crop`` for a random sized crop in efficientnet
        style.
        Args:
            img (ndarray): Image to be cropped.
            size (sequence): Desired output size of the crop.
            scale (tuple): Range of the random size of the cropped image
                compared to the original image size.
            ratio (tuple): Range of the random aspect ratio of the cropped
                image compared to the original image area.
            max_attempts (int): Maxinum number of attempts before falling back
                to central crop. Defaults to 10.
            min_covered (Number): Minimum ratio of the cropped area to the
                original area. Only valid if efficientnet_style is true.
                Defaults to 0.1.
            crop_padding (int): The crop padding parameter in efficientnet
                style center crop. Defaults to 32.
        Returns:
            tuple: Params (ymin, xmin, ymax, xmax) to be passed to `crop` for
                a random sized crop.
        """
        height, width = img.shape[:2]
        area = height * width
        min_target_area = scale[0] * area
        max_target_area = scale[1] * area

        for _ in range(max_attempts):
            aspect_ratio = random.uniform(*ratio)
            min_target_height = int(
                round(math.sqrt(min_target_area / aspect_ratio)))
            max_target_height = int(
                round(math.sqrt(max_target_area / aspect_ratio)))

            if max_target_height * aspect_ratio > width:
                max_target_height = int((width + 0.5 - 1e-7) / aspect_ratio)
                if max_target_height * aspect_ratio > width:
                    max_target_height -= 1

            max_target_height = min(max_target_height, height)
            min_target_height = min(max_target_height, min_target_height)

            # slightly differs from tf inplementation
            target_height = int(
                round(random.uniform(min_target_height, max_target_height)))
            target_width = int(round(target_height * aspect_ratio))
            target_area = target_height * target_width

            # slight differs from tf. In tf, if target_area > max_target_area,
            # area will be recalculated
            if (target_area < min_target_area or target_area > max_target_area
                    or target_width > width or target_height > height
                    or target_area < min_covered * area):
                continue

            ymin = random.randint(0, height - target_height)
            xmin = random.randint(0, width - target_width)
            ymax = ymin + target_height - 1
            xmax = xmin + target_width - 1

            return ymin, xmin, ymax, xmax

        # Fallback to central crop
        img_short = min(height, width)
        crop_size = size[0] / (size[0] + crop_padding) * img_short

        ymin = max(0, int(round((height - crop_size) / 2.)))
        xmin = max(0, int(round((width - crop_size) / 2.)))
        ymax = min(height, ymin + crop_size) - 1
        xmax = min(width, xmin + crop_size) - 1

        return ymin, xmin, ymax, xmax

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            img = results[key]
            if self.efficientnet_style:
                get_params_func = self.get_params_efficientnet_style
                get_params_args = dict(
                    img=img,
                    size=self.size,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts,
                    min_covered=self.min_covered,
                    crop_padding=self.crop_padding)
            else:
                get_params_func = self.get_params
                get_params_args = dict(
                    img=img,
                    scale=self.scale,
                    ratio=self.ratio,
                    max_attempts=self.max_attempts)
            ymin, xmin, ymax, xmax = get_params_func(**get_params_args)
            img = mmcv.imcrop(img, bboxes=np.array([xmin, ymin, xmax, ymax]))
            results[key] = mmcv.imresize(
                img,
                tuple(self.size[::-1]),
                interpolation=self.interpolation,
                backend=self.backend)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(size={self.size}'
        repr_str += f', scale={tuple(round(s, 4) for s in self.scale)}'
        repr_str += f', ratio={tuple(round(r, 4) for r in self.ratio)}'
        repr_str += f', max_attempts={self.max_attempts}'
        repr_str += f', efficientnet_style={self.efficientnet_style}'
        repr_str += f', min_covered={self.min_covered}'
        repr_str += f', crop_padding={self.crop_padding}'
        repr_str += f', interpolation={self.interpolation}'
        repr_str += f', backend={self.backend})'
        return repr_str



@PIPELINES.register_module()
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.trans = _transforms.RandomHorizontalFlip(p=p)

    def __call__(self, input_dict):
        img = input_dict["img"]
        img = Image.fromarray(img)
        img = self.trans(img)
        img = np.array(img)
        input_dict["img"] = img
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float): Probability.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# custom transforms
# @PIPELINES.register_module()
# class Lighting(object):
#     """Lighting noise(AlexNet - style PCA - based noise)."""

#     _IMAGENET_PCA = {
#         'eigval':
#         torch.Tensor([0.2175, 0.0188, 0.0045]),
#         'eigvec':
#         torch.Tensor([
#             [-0.5675, 0.7192, 0.4009],
#             [-0.5808, -0.0045, -0.8140],
#             [-0.5836, -0.6948, 0.4203],
#         ])
#     }

#     def __init__(self):
#         self.alphastd = 0.1
#         self.eigval = self._IMAGENET_PCA['eigval']
#         self.eigvec = self._IMAGENET_PCA['eigvec']

#     def __call__(self, img):
#         assert isinstance(img, torch.Tensor), \
#             "Expect torch.Tensor, got {}".format(type(img))
#         if self.alphastd == 0:
#             return img

#         alpha = img.new().resize_(3).normal_(0, self.alphastd)
#         rgb = self.eigvec.type_as(img).clone()\
#             .mul(alpha.view(1, 3).expand(3, 3))\
#             .mul(self.eigval.view(1, 3).expand(3, 3))\
#             .sum(1).squeeze()

#         return img.add(rgb.view(3, 1, 1).expand_as(img))

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str


@PIPELINES.register_module()
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, input_dict):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = input_dict["img"]
        img = Image.fromarray(img)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        img = np.array(img)
        input_dict["img"] = img
        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


# @PIPELINES.register_module()
# class Solarization(object):
#     """Solarization augmentation in BYOL https://arxiv.org/abs/2006.07733."""

#     def __init__(self, threshold=128):
#         self.threshold = threshold

#     def __call__(self, img):
#         img = np.array(img)
#         img = np.where(img < self.threshold, img, 255 -img)
#         return Image.fromarray(img.astype(np.uint8))

#     def __repr__(self):
#         repr_str = self.__class__.__name__
#         return repr_str