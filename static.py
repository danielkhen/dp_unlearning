import warnings
warnings.simplefilter("ignore")

import torch
import timm

from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Loader parameters
BATCH_SIZE = 128
NUM_WORKERS = 2

BASE_MODEL = VisionTransformer(
    img_size=32,                  # Set custom input size
    patch_size=4,                 # Set patch size
    embed_dim=192,                # Embedding dimension (DeiT-Tiny default)
    depth=12,                     # Number of transformer layers
    num_heads=3,                  # Number of attention heads
    mlp_ratio=4,                  # MLP hidden dimension ratio
    num_classes=10,               # Number of output classes (e.g., CIFAR-10)
    qkv_bias=True,                # Bias in QKV projections
    norm_layer=torch.nn.LayerNorm  # Layer normalization
)

#BASE_MODEL = timm.create_model('resnet18')

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

# Training parameters
LR = 3e-5
MAX_EPOCHS = 100
MIN_DELTA = 1e-5
PATIENCE = 300
CHECKPOINT_EVERY = 5
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
GRAD_ACCUMULATION_STEPS = 1
ALPHA = 0.9

# Differential privacy parameters
DELTA = 1e-5
EPSILON = 50.0
MAX_GRAD_NORM = 1.2

# LORA parameters
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 1e-1