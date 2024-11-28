import torch
import os
import static
import copy

from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import datasets
from opacus.validators import ModuleValidator

def load_dataset(dataset, dataset_transform, test_transform):
    dataset = getattr(datasets, dataset)

    # Download dataset if not already downloaded
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=dataset_transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # Load dataset
    trainloader = DataLoader(trainset, batch_size=static.BATCH_SIZE, num_workers=static.NUM_WORKERS, shuffle=True)
    testloader = DataLoader(testset, batch_size=static.BATCH_SIZE, num_workers=static.NUM_WORKERS, shuffle=False)

    return trainloader, testloader

def load_model(base_model, device = None, weights_from = None, fix_dp = True):
    # Define the model
    model = copy.deepcopy(base_model)
    
    if fix_dp:
        model = ModuleValidator.fix(model) # This replaces BatchNorm with GroupNorm

    if device: # Move to device if specified
        model.to(device)

    if weights_from and os.path.exists(weights_from): # Load pre-trained weights if specified
        model.load_state_dict(torch.load(weights_from, weights_only=True))

    return model
