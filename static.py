import warnings
warnings.simplefilter("ignore")

import torch
from torchvision import transforms

torch.manual_seed(69)

CPU = torch.device('cpu')
CUDA = torch.device('cuda')

# Dataset parameters
DATASET_NAME = 'CIFAR10'
CLASSES_NUM = 10
IMG_SIZE = 32
DATASET_SIZE = 50000

DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD = (0.2023, 0.1994, 0.2010)

AUGMENTATIONS = (
        # transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
)

NORMALIZATIONS = (
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
)
