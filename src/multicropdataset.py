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


class MultiCropDataset(datasets.CIFAR100):
    def __init__(self, data_path, size_crops, nmb_crops,
                 size_dataset=-1, return_index=False):
        super(MultiCropDataset, self).__init__(data_path, download=True, train=True)
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

            cutout = Cutout(4, 0.5)

            trans.extend([transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                crop,
                transforms.ToTensor(),
                normalize,
                cutout,
                ])] * nmb_crops[i])
        self.trans = trans


    def __getitem__(self, index):
        #path, _ = self.samples[index]
        #image = self.loader(path)
        image, _ = super().__getitem__(index)
        # image, _ = self.__getitem__(index)
        multi_crops = list(map(lambda trans: trans(image), self.trans))
        if self.return_index:
            return index, multi_crops
        return multi_crops


class Cutout:
    def __init__(self, mask_size, p):
        self.mask_size = mask_size
        self.p = p # probability we do the cutout

    def __call__(self, x):
        # First pick a random int to start for x and y
        if np.random.random() > self.p:
            return x

        xmin = ymin = -self.mask_size
        xmax = x.shape[1]
        ymax = x.shape[2]
        xstart = np.random.randint(xmin, xmax)
        xend = xstart + self.mask_size

        ystart = np.random.randint(ymin, ymax)
        yend = ystart + self.mask_size

        x[:, max(0, xstart):xend, max(0, ystart):yend].zero_()
        return x
