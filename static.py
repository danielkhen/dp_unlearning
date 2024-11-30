import warnings
warnings.simplefilter("ignore")

import torch

from more_itertools import powerset
from torchvision import transforms


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset parameters
DATASET_NAME = 'CIFAR10'
CLASSES_NUM = 10
IMG_SIZE = 32
DATASET_SIZE = 50000
TESTSET_SIZE = 10000

DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD = (0.2023, 0.1994, 0.2010)

AUGMENTATIONS = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
]

DP_AUGMENTATIONS = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=1),
]

NORMALIZATIONS = [
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
]

AUGMENTATION_TRANSFORM = transforms.Compose(AUGMENTATIONS + NORMALIZATIONS)
TRANSFORM = transforms.Compose(NORMALIZATIONS)
DP_TRANSFORMS = [transforms.Compose(subset) for subset in powerset(DP_AUGMENTATIONS)]