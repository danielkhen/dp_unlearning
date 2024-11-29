import argparse
import loader
import static
import trainer
import tester
import fine_tuning

from torch import nn, optim

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('model', type=str, help='model architecture name')
    parser.add_argument('output', type=str, help='path of pth file to save model weights to')

    parser.add_argument('--learning-rate', '--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--batch-size', '--bs', default=128, type=bool, help='batch size')
    parser.add_argument('--data-augmentation', '--da', action='store_true', help='data augmentation')
    parser.add_argument('--pretrained', '-p', action='store_true', help='wether model comes with pre-trained weights')
    parser.add_argument('--epochs', '-e', default=200, type=int, help='number of epochs')
    parser.add_argument('--optimizer', '-o', default='AdamW', type=str, help='optimizer to use from torch.nn.optim')
    parser.add_argument('--input-weights', '-i', default=None, type=str, help='path of pth file for pre-trained weights')
    parser.add_argument('--test', '-t', action='store_true', help='wether to only test the loaded model')

    parser.add_argument('--weight-decay', default=5e-2, type=float, help='weight decay used in optimizer')
    parser.add_argument('--num-workers', default=4, type=int, help='number of workers (dataset download)')
    parser.add_argument('--checkpoint-model', default=10, type=int, help='number of epochs to checkpoint the model')
    parser.add_argument('--accumulation-steps', default=1, type=int, help='number of steps to accumulate gradients')

    parser.add_argument('--differential-privacy', '--dp', default=None, type=str, choices=('fastdp', 'opacus'), help='wether to train the model with differential privacy')
    parser.add_argument('--epsilon', default=50.0, type=float, help='epsilon for differential privacy')
    parser.add_argument('--delta', default=1e-5, type=float, help='delta for differential privacy')
    parser.add_argument('--max-grad-norm', default=1.2, type=float, help='maximum gradient norm for differential privacy')

    parser.add_argument('--peft', default=None, type=str, choices=('lora', 'prune', 'prune-grads'), help='the peft method to use, either lora, prune or prune-grads')
    parser.add_argument('--peft-targets', nargs='*', type=str, help='list of target model children to apply peft on')
    parser.add_argument('--prune-amount', default=0.8, type=float, help='amount to prune in percentage of targets weights')
    parser.add_argument('--lora-rank', default=32, type=int, help='rank for LoRA')
    parser.add_argument('--lora-alpha', default=32, type=int, help='alpha for LoRA')
    parser.add_argument('--lora-dropout', default=1e-1, type=float, help='dropout for LoRA')

    args = parser.parse_args()

    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, static.AUGMENTATION_TRANSFORM if args.data_augmentation else static.TRANSFORM, static.TRANSFORM, args.batch_size, args.num_workers)

    model = loader.model_factory(args.model, weights_path=args.input_weights, fix_dp=True, pretrained=args.pretrained)

    criterion =  nn.CrossEntropyLoss()
        
    if args.test:
        return

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

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    match args.differential_privacy:
        case 'fastdp':
            from fastDP import PrivacyEngine
            
            privacy_engine = PrivacyEngine(
                model,
                batch_size=args.batch_size,
                sample_size=static.DATASET_SIZE,
                epochs=args.epochs,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm
            )

            privacy_engine.attach(optimizer)
        case 'opacus':
            from opacus import PrivacyEngine

            privacy_engine = PrivacyEngine()

            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                epochs=args.epochs,
                target_epsilon=args.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.max_grad_norm
            )

    trainer.train(model, train_loader, test_loader, criterion, optimizer, scheduler, args.output,
                epochs=args.epochs, accumulation_steps=args.accumulation_steps, checkpoint_model=args.checkpoint_model, state_dict={'args': args})

if __name__ == "__main__":
    main()