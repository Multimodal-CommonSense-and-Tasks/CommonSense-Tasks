from pathlib import Path
from random import randint, choice

import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, FakeData, VisionDataset
from pytorch_lightning import LightningDataModule
import torch
from typing import Any, Callable, Optional, Tuple
from torchvision import transforms
from PIL import Image
from io import BytesIO

import re
from tqdm import tqdm
import os
import csv
import numpy as np
import nltk
import copy
import random

#To prevent truncated error
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def identity(x):
    return x

class Grayscale2RGB:
    def __init__(self):  
        pass  
    def __call__(self, img):
        if img.mode != 'RGB':
            return img.convert('RGB') 
        else:
            return img
    def __repr__(self):
        return self.__class__.__name__ + '()'        


class ImageDataModule(LightningDataModule):

    def __init__(self, train_dir, val_dir, batch_size, num_workers, img_size, resize_ratio=0.75, 
                world_size = 1, dataset_size = [int(1e9)]):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size

        if len(dataset_size) == 1:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[0]
        else:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[1] 

        self.world_size = world_size
        self.transform_train = T.Compose([
                                    Grayscale2RGB(),
                                    T.RandomResizedCrop(img_size, scale=(resize_ratio, 1.),ratio=(1., 1.)),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                            
                                    ])
        self.transform_val = T.Compose([
                                    Grayscale2RGB(),
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                                
                                    ])
    def imagetransform(self, b):
        return Image.open(BytesIO(b))

    def dummy(self, s):
        return torch.zeros(1)

    def setup(self, stage=None):
        self.train_dataset = ImageDataset(self.train_dir, self.transform_train)
        self.val_dataset = ImageDataset(self.val_dir, self.transform_val)
  
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)





IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

# For Pororo dataset loader
class ImageDataset(ImageFolder):
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        return self.random_sample()

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        try:
            path, target = self.samples[index]
            sample = self.sample_image(self.loader(path))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {path}.")
            print(f"Skipping index {index}")
            return self.skip_sample(index)     
                   
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
    
class TextImageDataset(Dataset):
    def __init__(self,
                 folder,
                 text_len=256,
                 image_size=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 transform=None,
                 shuffle=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()
        self.shuffle = shuffle
        path = Path(folder)

        text_files = [*path.glob('**/*.txt')]
        image_files = [
            *path.glob('**/*.png'), *path.glob('**/*.jpg'),
            *path.glob('**/*.jpeg'), *path.glob('**/*.bmp')
        ]

        text_files = {text_file.stem: text_file for text_file in text_files}
        image_files = {image_file.stem: image_file for image_file in image_files}

        keys = (image_files.keys() & text_files.keys())

        self.keys = list(keys)
        self.text_files = {k: v for k, v in text_files.items() if k in keys}
        self.image_files = {k: v for k, v in image_files.items() if k in keys}
        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transform

    def __len__(self):
        return len(self.keys)
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)
        
    def __getitem__(self, ind):
        try:
            key = self.keys[ind]
            text_file = self.text_files[key]
            image_file = self.image_files[key]
            descriptions = text_file.read_text().split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))            
            description = choice(descriptions)
        except IndexError as zero_captions_in_file_ex:
            print(f"An exception occurred trying to load file {text_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)
        #return only image with image first order
        if self.text_len == 0:
            try:
                image_tensor = self.image_transform(PIL.Image.open(image_file))
            except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
                print(f"An exception occurred trying to load file {image_file}.")
                print(f"Skipping index {ind}")
                return self.skip_sample(ind)      
            return image_tensor, None     

        tokenized_text = self.tokenizer.tokenize(
            description,
            self.text_len,
            truncate_text=self.truncate_captions
        ).squeeze(0)
        try:
            image_tensor = self.image_transform(PIL.Image.open(image_file))
        except (PIL.UnidentifiedImageError, OSError) as corrupt_image_exceptions:
            print(f"An exception occurred trying to load file {image_file}.")
            print(f"Skipping index {ind}")
            return self.skip_sample(ind)

        # Success
        return tokenized_text, image_tensor


class DiscTextImageDataset(Dataset):
    def __init__(self,
                 img_folder='dataset/ducogan_pororo',
                 min_len=4,
                 text_len=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 transform=None,
                 mode='train',
                 annotations_file='descriptions.csv',
                 frame_counter='frames_counter.npy',
                 cache_dir='dataset/ducogan_pororo',
                 load_images=False,
                 debug_mode=False,
                 tmp_flag='val'
                 ):
                 
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        self.tmp_flag = tmp_flag

        self.img_folder = img_folder
        self.frame_counter_path = os.path.join(img_folder, frame_counter)
        self.annotation_path = os.path.join(img_folder, annotations_file)
        self.label_path = os.path.join(img_folder, 'labels.npy')

        self.mode = mode
        self.debug_mode = debug_mode
        self.img_dataset = ImageFolder(img_folder)

        self.images = []
        self.followings = []

        self.frame_counter = np.load(self.frame_counter_path, allow_pickle=True).item() 
        self.labels = np.load(self.label_path, allow_pickle=True, encoding='latin1').item()        


        if cache_dir is not None and os.path.exists(os.path.join(cache_dir, 'img_cache_inf.npy')):
            self.images = np.load(os.path.join(cache_dir, 'img_cache_inf.npy'), encoding='latin1')

        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc='Counting total number of frames')):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(img_folder, '')

                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))

                v_name = re.sub(r"[0-9]+.png",'', v_name)

                if v_name[0] == '/':
                    v_name = v_name[1:]

                self.images.append(img_path.replace(img_folder, ''))

            np.save(os.path.join(img_folder, 'img_cache_inf.npy'), self.images)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        self.train_ids = train_ids # 10191
        self.val_ids = val_ids     # 2334
        self.test_ids = test_ids   # 2208

        self.annotations = {}
        with open(self.annotation_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                episode_name, frame_id, caption = row
                self.annotations[os.path.join(episode_name, frame_id + '.png')] = caption

        print("Obtaining caption lengths...")
        all_tokens = [nltk.tokenize.word_tokenize(str(
            self.annotations[os.path.join(self.img_dataset.imgs[idx][0].split('/')[-2],
                                          self.img_dataset.imgs[idx][0].split('/')[-1])]
        ).lower()) for idx in tqdm(train_ids)]
        self.caption_lengths = [len(token) for token in all_tokens]
        self.max_t_len = max(self.caption_lengths) + 2
        print("Maximum caption length {}".format(self.max_t_len))

        if self.mode == 'train':
            self.ids = self.train_ids
            if self.debug_mode:
                self.ids = self.ids[:20] # debug mode
        elif self.mode =='val':
            self.ids = self.val_ids
            if self.debug_mode:
                self.ids = self.ids[:10]
        else:   # test
            self.ids = self.test_ids
        
        print("Total number of clips {}".format(len(self.ids)))

        if load_images:
            self.image_arrays = {}
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Loading all images")):
                img_path, _ = self.img_dataset.imgs[idx]
                self.image_arrays[img_path] = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')
        self.load_images = load_images
        self.pred_img_dir = None

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transform

    def __len__(self):
        return len(self.ids)
        
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
        
    def __getitem__(self, ind):

        # Obtain image and caption if in training mode

        data_id = self.ids[ind]
        image_seq_paths = [str(self.images[data_id])]
        
        ann_ids = [os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]) for img_path in image_seq_paths]
        raw_captions = [self.annotations[ann_id] for ann_id in ann_ids]
        
        # Convert image to tensor and pre-process using transform

        if self.load_images:
            images = [self.image_arrays[img_path] for img_path in image_seq_paths]
        else:
            if self.pred_img_dir:
                images = None
            else:
                images = []
                for img_path in image_seq_paths:
                    if img_path[0] == '/':
                        img_path = img_path[1:]
                    file_full_path = os.path.join(self.img_folder, img_path)
                    images.append(self.sample_image(Image.open(file_full_path).convert('RGB')))

        disc_features = []
        data = {}
        for i, raw_caption in enumerate(raw_captions):
            data['image'] = self.image_transform(images[i])
            tokenized_text = self.tokenizer.tokenize(raw_caption,
                                                     self.text_len,
                                                     truncate_text=self.truncate_captions
                                                    ).squeeze(0)

            data['tokenized_text'] = tokenized_text
            tmp_data = copy.deepcopy(data)
            disc_features.append(tmp_data)
            del tmp_data

        return disc_features


class DiscValTextImageDataset(Dataset):
    def __init__(self,
                 img_folder='dataset/ducogan_pororo',
                 min_len=4,
                 text_len=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 transform=None,
                 mode='train',
                 annotations_file='descriptions.csv',
                 frame_counter='frames_counter.npy',
                 cache_dir='dataset/ducogan_pororo',
                 load_images=False,
                 debug_mode=False,
                 tmp_flag='val'
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        self.tmp_flag = tmp_flag

        self.img_folder = img_folder
        self.frame_counter_path = os.path.join(img_folder, frame_counter)
        self.annotation_path = os.path.join(img_folder, annotations_file)
        self.label_path = os.path.join(img_folder, 'labels.npy')

        self.mode = mode
        self.debug_mode = debug_mode
        self.img_dataset = ImageFolder(img_folder)

        self.images = []
        self.followings = []

        self.frame_counter = np.load(self.frame_counter_path, allow_pickle=True).item() 
        self.labels = np.load(self.label_path, allow_pickle=True, encoding='latin1').item()      

        if cache_dir is not None and \
                os.path.exists(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy')) and \
                os.path.exists(os.path.join(cache_dir, 'following_cache' + str(min_len) +  '.npy')):
            self.images = np.load(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(cache_dir, 'following_cache' + str(min_len) + '.npy'))

        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc='Counting total number of frames')):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(img_folder, '')

                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))

                v_name = re.sub(r"[0-9]+.png",'', v_name)

                if v_name[0] == '/':
                    v_name = v_name[1:]
                if id > self.frame_counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(img_folder, ''))
                self.followings.append(following_imgs)
            np.save(img_folder + 'img_cache' + str(min_len) + '.npy', self.images)
            np.save(img_folder + 'following_cache' + str(min_len) + '.npy', self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)

        self.train_ids = train_ids # 10191
        self.val_ids = val_ids     # 2334
        self.test_ids = test_ids   # 2208

        self.annotations = {}
        with open(self.annotation_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                episode_name, frame_id, caption = row
                self.annotations[os.path.join(episode_name, frame_id + '.png')] = caption

        print("Obtaining caption lengths...")
        all_tokens = [nltk.tokenize.word_tokenize(str(
            self.annotations[os.path.join(self.img_dataset.imgs[idx][0].split('/')[-2],
                                          self.img_dataset.imgs[idx][0].split('/')[-1])]
        ).lower()) for idx in tqdm(train_ids)]

        self.caption_lengths = [len(token) for token in all_tokens]
        self.max_t_len = max(self.caption_lengths) + 2
        print("Maximum caption length {}".format(self.max_t_len))

        if self.mode == 'train':
            self.ids = self.train_ids
            if self.debug_mode:
                self.ids = self.ids[:20] # debug mode
        elif self.mode =='val':
            self.ids = self.val_ids
            if self.debug_mode:
                self.ids = self.ids[:10]
        else:   # test
            self.ids = self.test_ids
        
        print("Total number of clips {}".format(len(self.ids)))

        if load_images:
            self.image_arrays = {}
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Loading all images")):
                img_path, _ = self.img_dataset.imgs[idx]
                self.image_arrays[img_path] = Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')
        self.load_images = load_images
        self.pred_img_dir = None

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transform

    def __len__(self):
        return len(self.ids)
        
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))
        
    def __getitem__(self, ind):

        # Obtain image and caption if in training mode
        data_id = self.ids[ind]
        image_seq_paths = [self.images[data_id].decode('utf-8')]
        
        for img_file in self.followings[data_id]:
            image_seq_paths.append(img_file.decode('utf-8'))
        ann_ids = [os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]) for img_path in image_seq_paths]
        raw_captions = [self.annotations[ann_id] for ann_id in ann_ids]

        # Convert image to tensor and pre-process using transform
        if self.load_images:
            images = [self.image_arrays[img_path] for img_path in image_seq_paths]
        else:
            if self.pred_img_dir:
                images = None
            else:
                images = [self.sample_image(Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')) for img_path in image_seq_paths]


        story_features = []
        data = {}
        for i, raw_caption in enumerate(raw_captions):
            data['image'] = self.image_transform(images[i])
            tokenized_text = self.tokenizer.tokenize(raw_caption, self.text_len, truncate_text=self.truncate_captions).squeeze(0)

            data['tokenized_text'] = tokenized_text
            data['idx'] = data_id
            tmp_data = copy.deepcopy(data)
            story_features.append(tmp_data)
            del tmp_data

        return story_features


class StoryTextImageDataModule(LightningDataModule):

    def __init__(self, data_dir, batch_size, num_workers, img_size, text_seq_len, 
                resize_ratio=0.75, truncate_captions=True, tokenizer=None, 
                world_size = 1, dataset_size = [int(1e9)], load_images=False, debug_mode=False):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.text_seq_len = text_seq_len
        self.resize_ratio = resize_ratio
        self.truncate_captions = truncate_captions
        self.tokenizer = tokenizer
        self.world_size = world_size
        self.load_images = load_images
        self.debug_mode = debug_mode

        if len(dataset_size) == 1:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[0]
        else:
            self.train_dataset_size = dataset_size[0]  
            self.val_dataset_size = dataset_size[1]    

        self.truncate_captions = truncate_captions
        self.transform_train = T.Compose([
                                    Grayscale2RGB(),   
                                    T.RandomResizedCrop(img_size,
                                            scale=(resize_ratio, 1.),ratio=(1., 1.)),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                            
                                    ])
        self.transform_val = T.Compose([
                                    Grayscale2RGB(),  
                                    T.Resize(img_size),
                                    T.CenterCrop(img_size),
                                    T.ToTensor(),
                                    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                                
                                    ])
    def imagetransform(self, b):
        return Image.open(BytesIO(b))

    def tokenize(self, s):
        if self.tokenizer == None:
            return None
        else:
            return self.tokenizer.tokenize(
                s.decode('utf-8'),
                self.text_seq_len,
                truncate_text=self.truncate_captions).squeeze(0) 

    def setup(self, stage=None):
        #for VAE training and DALLE training
        self.train_dataset = StoryTextImageDataset(
                                img_folder=self.data_dir,
                                text_len=self.text_seq_len,
                                resize_ratio=self.resize_ratio,
                                truncate_captions=self.truncate_captions,
                                tokenizer=self.tokenizer,
                                transform=self.transform_train,
                                mode='train',
                                load_images=self.load_images,
                                debug_mode=self.debug_mode
                                )
        self.val_dataset = StoryTextImageDataset(
                                img_folder=self.data_dir,
                                text_len=self.text_seq_len,
                                resize_ratio=self.resize_ratio,
                                truncate_captions=self.truncate_captions,
                                tokenizer=self.tokenizer,
                                transform=self.transform_val,
                                mode='val',
                                load_images=self.load_images,
                                debug_mode=self.debug_mode
                                )
        self.test_dataset = StoryTextImageDataset(
                                img_folder=self.data_dir,
                                text_len=self.text_seq_len,
                                resize_ratio=self.resize_ratio,
                                truncate_captions=self.truncate_captions,
                                tokenizer=self.tokenizer,
                                transform=self.transform_val,
                                mode='test',
                                load_images=self.load_images,
                                debug_mode=self.debug_mode
                                )
                

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class StoryTextImageDataset(Dataset):
    def __init__(self,
                 img_folder='dataset/ducogan_pororo',
                 min_len=4,
                 text_len=256,
                 truncate_captions=False,
                 resize_ratio=0.75,
                 tokenizer=None,
                 transform=None,
                 mode='train',
                 annotations_file='descriptions.csv',
                 frame_counter='frames_counter.npy',
                 cache_dir='dataset/ducogan_pororo',
                 load_images=False,
                 debug_mode=False
                 ):
        """
        @param folder: Folder containing images and text files matched by their paths' respective "stem"
        @param truncate_captions: Rather than throw an exception, captions which are too long will be truncated.
        """
        super().__init__()

        self.img_folder = img_folder
        self.frame_counter_path = os.path.join(img_folder, frame_counter)
        self.annotation_path = os.path.join(img_folder, annotations_file)
        self.label_path = os.path.join(img_folder, 'labels.npy')

        self.mode = mode
        self.debug_mode = debug_mode
        self.img_dataset = ImageFolder(img_folder)
        
        self.images = []
        self.followings = []

        self.frame_counter = np.load(self.frame_counter_path, allow_pickle=True).item() 
        self.labels = np.load(self.label_path, allow_pickle=True, encoding='latin1').item()
        
        
        if cache_dir is not None and \
                os.path.exists(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy')) and \
                os.path.exists(os.path.join(cache_dir, 'following_cache' + str(min_len) +  '.npy')):
            self.images = np.load(os.path.join(cache_dir, 'img_cache' + str(min_len) + '.npy'), encoding='latin1')
            self.followings = np.load(os.path.join(cache_dir, 'following_cache' + str(min_len) + '.npy'))

        else:
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc='Counting total number of frames')):
                img_path, _ = self.img_dataset.imgs[idx]
                v_name = img_path.replace(img_folder, '')

                id = v_name.split('/')[-1]
                id = int(id.replace('.png', ''))

                v_name = re.sub(r"[0-9]+.png",'', v_name)

                if v_name[0] == '/':
                    v_name = v_name[1:]
                if id > self.frame_counter[v_name] - min_len:
                    continue
                following_imgs = []
                for i in range(min_len):
                    following_imgs.append(v_name + str(id+i+1) + '.png')
                self.images.append(img_path.replace(img_folder, ''))
                self.followings.append(following_imgs)
            np.save(img_folder + 'img_cache' + str(min_len) + '.npy', self.images)
            np.save(img_folder + 'following_cache' + str(min_len) + '.npy', self.followings)

        train_ids, val_ids, test_ids = np.load(os.path.join(img_folder, 'train_seen_unseen_ids.npy'), allow_pickle=True)
        self.train_ids = train_ids # 10191
        self.val_ids = val_ids     # 2334
        self.test_ids = test_ids   # 2208

        self.annotations = {}
        with open(self.annotation_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                episode_name, frame_id, caption = row
                self.annotations[os.path.join(episode_name, frame_id + '.png')] = caption

        print("Obtaining caption lengths...")
        all_tokens = [nltk.tokenize.word_tokenize(str(
            self.annotations[os.path.join(self.img_dataset.imgs[idx][0].split('/')[-2],
                                          self.img_dataset.imgs[idx][0].split('/')[-1])]
        ).lower()) for idx in tqdm(train_ids)]
        self.caption_lengths = [len(token) for token in all_tokens]
        self.max_t_len = max(self.caption_lengths) + 2
        print("Maximum caption length {}".format(self.max_t_len))

        if self.mode == 'train':
            self.ids = self.train_ids
            if self.debug_mode:
                self.ids = self.ids[:20] # debug mode
        elif self.mode =='val':
            self.ids = self.val_ids
            if self.debug_mode:
                self.ids = self.ids[:10]
        else:   # test
            self.ids = self.test_ids
            if self.debug_mode:
                self.ids = self.ids[:300]
            
        
        print("Total number of clips {}".format(len(self.ids)))

        if load_images:
            self.image_arrays = {}
            for idx, (im, _) in enumerate(tqdm(self.img_dataset, desc="Loading all images")):
                img_path, _ = self.img_dataset.imgs[idx]
                self.image_arrays[img_path] = Image.open(img_path).convert('RGB')
        self.load_images = load_images
        self.pred_img_dir = None

        self.text_len = text_len
        self.truncate_captions = truncate_captions
        self.resize_ratio = resize_ratio
        self.tokenizer = tokenizer
        self.image_transform = transform

    def __len__(self):
        return len(self.ids)
        
    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def sample_image(self, im):
        shorter, longer = min(im.size[0], im.size[1]), max(im.size[0], im.size[1])
        video_len = int(longer/shorter)
        se = np.random.randint(0,video_len, 1)[0]
        return im.crop((0, se * shorter, shorter, (se+1)*shorter))

    def sample_sentence(self, sents):
        tmp_sents = []

        if len(sents.split('. ')) > 1:
            for i, tmp_sent in enumerate(sents.split('. ')):

                if (i+1) == len(sents.split('. ')):
                    tmp_sents.append(tmp_sent)
                else:
                    tmp_sents.append(tmp_sent + '.')

            return random.sample(tmp_sents, 1)[0]
        else:
            return sents
        
    def __getitem__(self, ind):

        # Obtain image and caption if in training mode
        data_id = self.ids[ind]
        image_seq_paths = [self.images[data_id].decode('utf-8')]
        
        for img_file in self.followings[data_id]:
            image_seq_paths.append(img_file.decode('utf-8'))
        ann_ids = [os.path.join(img_path.split('/')[-2], img_path.split('/')[-1]) for img_path in image_seq_paths]
        raw_captions = [self.annotations[ann_id] for ann_id in ann_ids]
        
        # Convert image to tensor and pre-process using transform
        if self.load_images:
            images = [self.image_arrays[img_path] for img_path in image_seq_paths]
            labels = [self.labels[img_path.split('.')[0]] for img_path in image_seq_paths]
        else:
            if self.pred_img_dir:
                images = None
            else:
                images = [self.sample_image(Image.open(os.path.join(self.img_folder, img_path)).convert('RGB')) for img_path in image_seq_paths]
                labels = [self.labels[img_path.split('.')[0]] for img_path in image_seq_paths]

        
        story_features = []
        data = {}
        for i, raw_caption in enumerate(raw_captions):
            data['image'] = self.image_transform(images[i])

            # raw_caption_splited = self.sample_sentence(raw_caption)
            tokenized_text = self.tokenizer.encode(raw_caption.lower())
            tokenized_text = torch.LongTensor(tokenized_text.ids)                   
                      

            data['tokenized_text'] = tokenized_text
            #data['tokenized_text_splited'] = tokenized_text_splited
            data['label'] = labels[i]
            data['idx'] = data_id
            tmp_data = copy.deepcopy(data)
            story_features.append(tmp_data)
            del tmp_data

        return story_features
