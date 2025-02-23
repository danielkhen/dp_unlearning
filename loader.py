import static
import torch
import timm
import math

from timm.models.vision_transformer import VisionTransformer
from torchvision import datasets
from opacus.validators import ModuleValidator
from torch.nn import LayerNorm
from torch.utils.data import Sampler, DataLoader
    
class MultiplicitySampler(Sampler):
    def __init__(self, dataset, augmentation_multiplicity):
        self.samples_count = len(dataset)
        self.augmentation_multiplicity = augmentation_multiplicity

    def __iter__(self):
        sample_indices = (torch.randperm(self.samples_count)).tolist()

        for sample_index in sample_indices:
            for _ in range(self.augmentation_multiplicity):
                yield sample_index

    def __len__(self):
        return self.samples_count * self.augmentation_multiplicity
    

def load_dataset(dataset, dataset_transform, testset_transform, batch_size, num_workers, augmentation_multiplicity=1, unlearning=False, forgetset_size=10000):
    dataset = getattr(datasets, dataset)

    # Download dataset if not already downloaded
    trainset = dataset(root='./data', train=True, download=True, transform=dataset_transform)
    testset = dataset(root='./data', train=False, download=True, transform=testset_transform)
    
    if unlearning:
        trainset, forgetset = torch.utils.data.random_split(trainset, (len(trainset) - forgetset_size, forgetset_size))

    # Load dataset
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, sampler=MultiplicitySampler(trainset, augmentation_multiplicity))
    forgetloader = DataLoader(forgetset, batch_size=batch_size, num_workers=num_workers, shuffle=MultiplicitySampler(forgetset, augmentation_multiplicity)) if unlearning else None
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, forgetloader, testloader

def model_factory(model_name, state_dict=None, differential_privacy=None, pretrained=False):
    match model_name:
        case 'vit-tiny':
            model = VisionTransformer(
                img_size=static.IMG_SIZE, 
                patch_size=4,
                embed_dim=192,
                depth=12, 
                num_heads=3, 
                mlp_ratio=4,
                num_classes=static.CLASSES_NUM,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case 'vit-small':
            model = VisionTransformer(
                img_size=static.IMG_SIZE, 
                patch_size=4,
                embed_dim=384,
                depth=12, 
                num_heads=6, 
                mlp_ratio=4,
                num_classes=static.CLASSES_NUM,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case 'vit-base':
            model = VisionTransformer(
                img_size=static.IMG_SIZE, 
                patch_size=4,
                embed_dim=768,
                depth=12, 
                num_heads=12, 
                mlp_ratio=4,
                num_classes=static.CLASSES_NUM,
                qkv_bias=True,
                norm_layer=LayerNorm
            )
        case _:
            model = timm.create_model(model_name, num_classes=10, pretrained=pretrained)

    if model_name in ('vit-base', 'vit-small', 'vit-tiny') and pretrained: # Load pretrained manually
        timm_name = model_name.replace('-', '_') + '_patch16_224'
        timm_model = timm.create_model(timm_name, pretrained=True)
        timm_state_dict = timm_model.state_dict()

        for param in ('pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias','head.weight', 'head.bias'):
            del timm_state_dict[param]

        state_dict = model.state_dict()
        state_dict.update(timm_state_dict)
        model.load_state_dict(state_dict)

    if differential_privacy == 'opacus':
        model = ModuleValidator.fix(model, weights_only=False)

    if state_dict:
        model.load_state_dict(state_dict)

    return model
