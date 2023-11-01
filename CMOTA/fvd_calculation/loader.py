# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example code that computes FVD for some empty frames.
The FVD for this setup should be around 131.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import functools
import PIL
import re
import pdb
from tqdm import tqdm
import random
import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

# Number of videos must be divisible by 16.
#NUMBER_OF_VIDEOS = 320
#VIDEO_LENGTH = 10

import numpy as np

def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

class VideoGenerateDataset(torch.utils.data.Dataset):
  def __init__(self, folder, min_len):
    self.folder = folder
    self.storys = []
    story = []

    tot_imgs = len(os.listdir(folder))
    for i in range(tot_imgs):
      i += 1
      story += [i]
      if i % min_len == 0:
        self.storys += [story]
        story = []
    print('Total number of clips: {}'.format(len(self.storys)))

  def __getitem__(self, item):
    # return a training list
    img_list = self.storys[item]
    images = []
    for img in img_list:
      #img = '{}.jpg'.format(img)
      img = '{}.png'.format(img)
      im = PIL.Image.open(os.path.join(self.folder, img))
      im = im.convert('RGB')
      im = center_crop(np.array(im), 224, 224)
      #im = im.resize((224, 224))
      #print(img.size)
      images.append(np.expand_dims(np.array(im), axis=0))
    images = np.concatenate(images, axis=0)
    return images

  def __len__(self):
    return len(self.storys)

class VideoGenerateInstanceDataset(torch.utils.data.Dataset):
  def __init__(self, folder, min_len):
    self.folder = folder
    self.storys = []
    story = []

    tot_imgs = len(os.listdir(folder))
    for i in range(tot_imgs):
      i += 1
      story += [i]
      if i % min_len == 0:
        self.storys += [story] * 256
        story = []
    print('Total number of clips: {}'.format(len(self.storys)))

  def __getitem__(self, item):
    # return a training list
    img_list = self.storys[item]
    images = []
    for img in img_list:
      img = '{}.png'.format(img)
      im = PIL.Image.open(os.path.join(self.folder, img))
      im = im.convert('RGB')
      im = center_crop(np.array(im), 224, 224)
      #im = im.resize((256,256))
      images.append(np.expand_dims(np.array(im), axis=0))
    images = np.concatenate(images, axis=0)
    return images

  def __len__(self):
    return len(self.storys)
