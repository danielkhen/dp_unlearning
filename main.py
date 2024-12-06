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
        state_dict = torch.load(args.inputs_weights, weights_only=True)
        model_state_dict = state_dict['model']
        print(f"Loading pretrained model with Test loss: {state_dict['loss']}, Test accuracy: {state_dict['accuracy']:.2f}")

    model = loader.model_factory(args.model, state_dict=model_state_dict if args.input_weights else None, fix_dp=True, pretrained=args.pretrained)

    if args.weight_standardization:
        modules.standardize_model(model)

    if args.peft:
        trainable_parameters = 0

        match args.peft:
            case 'lora':
                model, trainable_parameters = fine_tuning.get_lora_model(model, target_children=args.peft_targets, rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
            case 'prune':
                trainable_parameters = fine_tuning.prune_model(model, args.peft_targets, args.prune_amount)
            case 'prune-grads':
                trainable_parameters = fine_tuning.prune_gradients(model, args.peft_targets, args.prune_amount)

        print(f"Number of trainable parameters using PEFT method {args.peft}: {trainable_parameters}")

    optimizer_class = getattr(optim, args.optimizer)
    optimizer = optimizer_class(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion =  nn.CrossEntropyLoss()
    
    schedulers = []

    if args.cosine_anealing:
        schedulers.append(optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs))

    if args.exponential_moving_average:
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay), use_buffers=True, device=static.CPU)

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
            poisson_sampling=False
        )

    starting_state_dict={
        'args': args,
        'trained_on': {
            'args': state_dict['args'],
            'input_weights': args.input_weights
        } if args.input_weights else None
    }

    trainer.train(model, train_loader, test_loader, criterion, optimizer, args.output, schedulers=schedulers, epochs=args.epochs, 
                checkpoint_every=args.checkpoint_every, state_dict=starting_state_dict, differential_privacy=args.differential_privacy, 
                loss_goal=args.loss_goal, ma_model=ema_model if args.exponential_moving_average else None, max_physical_batch_size=args.max_physical_batch_size)

if __name__ == "__main__":
    main()