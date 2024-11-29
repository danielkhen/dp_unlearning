import warnings
warnings.simplefilter("ignore")

import torch
import timm

from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Dataset parameters
DATASET_NAME = 'CIFAR10'
CLASSES_NUM = 10
DATASET_SIZE = 50000
TESTSET_SIZE = 10000

DATASET_MEAN = (0.4914, 0.4822, 0.4465)
DATASET_STD = (0.2023, 0.1994, 0.2010)

AUGMENTATION_TRANSFORM = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
])

TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(DATASET_MEAN, DATASET_STD)
])