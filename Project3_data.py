import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import itertools
import random
import math
import os


def channel_norm(img):
    # img: ndarray, float32
    mean = np.mean(img)
    std = np.std(img)
    pixels = (img - mean) / (std + 0.0000001)
    return pixels


def parse_line(line):
    line_parts = line.strip().split(',')
    id = line_parts[0]
    path = line_parts[1]
    classes = line_parts[2]
    classes = np.long(classes)
    return id, path, classes


class Normalize(object):
    """
        Resize to train_boarder x train_boarder. Here we use 112 x 112
        Then do channel normalization: (image - mean) / std_variation
    """

    def __call__(self, sample):
        image, path, classes = sample['image'], sample['path'], sample["class"]
        w, h = image.size  # Image.size==>W*H
        train_boarder = 112
        image_resize = np.asarray(
            image.resize((train_boarder, train_boarder), Image.BILINEAR),
            dtype=np.float32)  # Image.ANTIALIAS)
        image = channel_norm(image_resize)

        return {'image': image,
                'path': path,
                'class': classes
                }


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        Tensors channel sequence: N x C x H x W
    """

    def __call__(self, sample):
        image, path, classes = sample['image'], sample['path'], sample["class"]
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # image = image.transpose((2, 0, 1))
        #image = np.expand_dims(image, axis=0)
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'path': path,
                'class': classes
                }


class RandomFlip(object):
    def __call__(self, sample):
        image, path, classes = sample['image'], sample['path'], sample["class"]
        # Flip image
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image,
                'path': path,
                'class': classes
                }


class RandomRotate(object):
    def __call__(self, sample):
        image, path, classes = sample['image'], sample['path'], sample["class"]
        # random rotate (-angle,angle)
        angle = 15
        a0 = random.uniform(-1, 1) * angle
        a1, a2 = a0, a0 * math.pi / 180
        ox, oy = image.width // 2, image.height // 2
        image = image.rotate(-a1, Image.BILINEAR, expand=0)
        return {'image': image,
                'path': path,
                'class': classes
                }


class FaceLandmarksDataset(Dataset):
    def __init__(self, src_lines, transform=None):
        self.lines = src_lines
        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        id, path, classes = parse_line(self.lines[idx])
        # image
        img = Image.open(path).convert('RGB')

        sample = {'image': img, 'path':path, 'class':classes}
        sample = self.transform(sample)
        #sample['class'] = int(classes)
        # if self.phase != 'train':
        #     sample['path'] = path
        #     sample['id'] = id
        return sample


def load_data(phase):
    data_file = phase + '_label' + '.csv'
    data_file = os.path.join('traffic-sign', data_file)
    with open(data_file) as f:
        lines = f.readlines()[1:]
    if phase == 'Train' or phase == 'train':
        tsfm = transforms.Compose([
            # RandomFlip(),
            RandomRotate(),
            Normalize(),  # do channel normalization
            ToTensor()  # convert to torch type: NxCxHxW
        ])
    else:
        tsfm = transforms.Compose([
            Normalize(),
            ToTensor()
        ])
    data_set = FaceLandmarksDataset(lines, transform=tsfm)
    return data_set


def get_train_test_set():
    train_set = load_data('train')
    valid_set = load_data('test')
    return train_set, valid_set


if __name__ == '__main__':
    train_set = load_data('train')
    indexes = np.random.randint(0, len(train_set), 6)
    fig = plt.figure(figsize=(30, 10))
    axes = fig.subplots(nrows=1, ncols=6)
    for i in range(6):
        sample = train_set[indexes[i]]
        ax = axes[i]
        img = sample['image']
        img = img[0]
        ax.imshow(img)
    plt.show()
