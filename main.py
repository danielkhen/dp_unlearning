import modules
import loader
import static
import trainer
import fine_tuning
import torch

from torch import nn, optim
from torchvision import transforms
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from opacus import PrivacyEngine
from parser import parser

def main():
    args = parser.parse_args()

    testset_transform = transforms.Compose(static.NORMALIZATIONS)
    dataset_transform = transforms.Compose(static.AUGMENTATIONS + static.NORMALIZATIONS) if args.data_augmentation or args.augmentation_multiplicity != 1 else testset_transform

    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, dataset_transform, testset_transform, args.batch_size, args.num_workers, 
                                                    augmentation_multiplicity=args.augmentation_multiplicity)
    
    if args.input_weights:
        state_dict = torch.load(args.input_weights)
        model_state_dict = state_dict['model']
        
        if not args.input_weights.endswith('.checkpoint'):
            print(f"Loading pretrained model with Test loss: {state_dict['loss']}, Test accuracy: {state_dict['accuracy']:.2f}")

    if args.test:
        return

    model = loader.model_factory(args.model, state_dict=model_state_dict if args.input_weights else None, fix_dp=not args.no_fix_dp, 
                                 pretrained=args.pretrained, fix_dp_kwargs=args.fix_dp_kwargs)

    if args.weight_standardization:
        modules.standardize_model(model)

    if args.peft:
        if args.peft_ratio:
            args.peft_ratio = args.peft_ratio * len(args.peft_targets)

        named_modules = dict(model.named_modules())
        peft_modules = tuple(getattr(nn, module) for module in args.peft_modules)                                   

        target_modules = [(f'{name}.{module_name}' if module_name else name, module, peft_ratio)
                            for name, peft_ratio in zip(args.peft_targets, args.peft_ratio)
                            for module_name, module in named_modules[name].named_modules()
                            if not peft_modules or isinstance(module, peft_modules)]
    
        match args.peft:
            case 'lora':
                model = fine_tuning.get_lora_model(model, target_modules, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            case 'prune':
                fine_tuning.prune(model, target_modules, importance=args.pruning_importance)
            case 'conv-adapter':
                for name, _, peft_ratio in target_modules:
                    fine_tuning.replace_module(model, name, modules.ParallelAdapter, args_lambda=lambda m: (m, modules.ConvAdapter),
                                                kwargs_lambda=lambda m: {
                                                    'inplanes': m.in_channels,
                                                    'outplanes': m.out_channels,
                                                    'peft_ratio': peft_ratio,
                                                    'kernel_size': m.kernel_size,
                                                    'stride': m.stride,
                                                    'padding': m.padding,
                                                    'weight_standardization': args.weight_standardization
                                                })
                    
            case 'linear-adapter':
                for name, _, peft_ratio in target_modules:
                    fine_tuning.replace_module(model, name, modules.ParallelAdapter, args_lambda=lambda m: (m, modules.LinearAdapter),
                                                kwargs_lambda=lambda m: {
                                                    'inplanes': m.in_features,
                                                    'outplanes': m.out_features,
                                                    'bias': m.bias,
                                                    'peft_ratio': peft_ratio,
                                                })
            case 'freeze':
                for _, module, _ in target_modules:
                    module.weight.requires_grad = False

        print(f"Number of trainable parameters using PEFT method {args.peft}: {sum(param.numel() for param in model.parameters() if param.requires_grad)}")

    optimizer_class = getattr(optim, args.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, **args.optimizer_kwargs)
    criterion =  nn.CrossEntropyLoss()
    
    schedulers = []

    if args.cosine_anealing:
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs))

    if args.exponential_moving_average:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True, device=static.CUDA)

    if args.differential_privacy:
        privacy_engine = PrivacyEngine()

        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=args.epochs,
            target_epsilon=args.epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
            grad_sample_mode=args.grad_sample_mode,
            poisson_sampling=False # Must be false so incomplete batches wouldn't be counted
        )

    starting_state_dict={
        'args': args,
        'trained_on': {
            'args': state_dict['args'],
            'input_weights': args.input_weights
        } if args.input_weights else None
    }

    model.to(static.CUDA)
    trainer.train(model, train_loader, test_loader, criterion, optimizer, args.output, schedulers=schedulers, epochs=args.epochs, 
                checkpoint_every=args.checkpoint_every, state_dict=starting_state_dict, differential_privacy=args.differential_privacy, 
                loss_goal=args.loss_goal, ma_model=ema_model if args.exponential_moving_average else None, max_physical_batch_size=args.max_physical_batch_size,
                augmentation_multiplicity=args.augmentation_multiplicity, grad_sample_mode=args.grad_sample_mode)

if __name__ == "__main__":
    main()