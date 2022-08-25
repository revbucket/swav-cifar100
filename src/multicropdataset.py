# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger

from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(self, data_path, size_crops, nmb_crops,
                 size_dataset=-1, return_index=False):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        # CIFAR XFORMS: Flip -> Crop -> Normalize -> Cutout
        xform_pipeline = []
        mean = [0.5071, 0.4865, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        normalize = transforms.Normalize(mean=mean, std=std)
        trans = []
        for i in range(len(size_crops)):
            crop = transforms.RandomCrop(
                size_crops[i], padding=2,)


            trans.extend([transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                crop,
                transforms.ToTensor(),
                normalize,
                cutout(4, 0.5, False)
                ])] * nmb_crops[i])
        self.trans = trans

        
    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops



def cutout(mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
    mask_size_half = mask_size // 2
    offset = 1 if mask_size % 2 == 0 else 0

    def _cutout(image):
        image = np.asarray(image).copy()

        if np.random.random() > p:
            return image

        h, w = image.shape[:2]

        if cutout_inside:
            cxmin, cxmax = mask_size_half, w + offset - mask_size_half
            cymin, cymax = mask_size_half, h + offset - mask_size_half
        else:
            cxmin, cxmax = 0, w + offset
            cymin, cymax = 0, h + offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - mask_size_half
        ymin = cy - mask_size_half
        xmax = xmin + mask_size
        ymax = ymin + mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        image[ymin:ymax, xmin:xmax] = mask_color
        return image

    return _cutout

