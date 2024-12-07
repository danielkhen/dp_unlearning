import static
import torch
import timm
import math

from timm.models.vision_transformer import VisionTransformer
from torchvision import datasets
from opacus.validators import ModuleValidator
from torch.nn import LayerNorm
from torch.utils.data import BatchSampler, DataLoader
    
class MultiplicityBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, augmentation_multiplicity):
        self.batch_size = batch_size
        self.augmentation_multiplicity = augmentation_multiplicity
        self.samples_per_batch = batch_size // augmentation_multiplicity
        self.samples_count = len(dataset)

    def __iter__(self):
        sample_indices = torch.randperm(self.samples_count)
        per_batch_indices = sample_indices.split(self.samples_per_batch)

        # Repeat each index augmentation multiplicity times
        for batch_indices in per_batch_indices:
            batch = batch_indices.repeat_interleave(self.augmentation_multiplicity)
            
            yield batch.tolist()

    def __len__(self):
        return math.ceil(self.samples_count / self.samples_per_batch)

def load_dataset(dataset, dataset_transform, testset_transform, batch_size, num_workers, augmentation_multiplicity=1):
    dataset = getattr(datasets, dataset)

    # Download dataset if not already downloaded
    trainset = dataset(root='./data', train=True, download=True, transform=dataset_transform)
    testset = dataset(root='./data', train=False, download=True, transform=testset_transform)

    # Random batch sampler for augmentation multiplicity
    batch_sampler = MultiplicityBatchSampler(trainset, batch_size, augmentation_multiplicity)

    # Load dataset
    trainloader = DataLoader(trainset, batch_sampler=batch_sampler, num_workers=num_workers)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, testloader

def model_factory(model_name, state_dict=None, fix_dp=True, pretrained=False):
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

    if fix_dp:
        model = ModuleValidator.fix(model, num_groups=16)

    if state_dict:
        model.load_state_dict(state_dict)

    model.to(static.CUDA)

    return model
