import static
import torch
import timm

from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import DataLoader
from torchvision import datasets
from opacus.validators import ModuleValidator
from torch.nn import LayerNorm

def load_dataset(dataset, dataset_transform, testset_transform, batch_size, num_workers):
    dataset = getattr(datasets, dataset)

    # Download dataset if not already downloaded
    trainset = dataset(root='./data', train=True, download=True, transform=dataset_transform)
    testset = dataset(root='./data', train=False, download=True, transform=testset_transform)

    # Load dataset
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    testloader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, testloader

def model_factory(model_name, weights_path=None, fix_dp=True, pretrained=False):
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

    if weights_path:
        state_dict = torch.load(weights_path, weights_only=True)
        model.load_state_dict(state_dict['model'])
        print(f"Loaded pretrained model with Test loss: {state_dict['loss']}, Test accuracy: {state_dict['accuracy']:.2f}")

    model.to(static.DEVICE)

    return model
