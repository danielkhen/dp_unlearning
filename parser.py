import argparse

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')

            if value.isdigit():
                value = int(value)
            elif value.replace('.', '', 1).isdigit():
                value = float(value)
            elif value[0] in ['"', '"'] and value[0] in ['"', '"']:
                value = value[1:-1]

            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('model', type=str, help='model architecture name')
parser.add_argument('output', type=str, help='path of pth file to save model weights to')

parser.add_argument('--test', '-t', action='store_true', help='only show previous training results for input model')
parser.add_argument('--learning-rate', '--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch-size', '--bs', default=128, type=int, help='batch size')
parser.add_argument('--data-augmentation', '--da', action='store_true', help='data augmentation')
parser.add_argument('--pretrained', '-p', action='store_true', help='wether model comes with pre-trained weights')
parser.add_argument('--epochs', '-e', default=200, type=int, help='number of epochs')
parser.add_argument('--optimizer', '-o', default='SGD', type=str, help='optimizer to use from torch.nn.optim')
parser.add_argument('--optimizer-kwargs', '-m', default={}, nargs='*', action=ParseKwargs, help='kwargs to use in optimizer')
parser.add_argument('--input-weights', '-i', default=None, type=str, help='path of pth file for pre-trained weights')
parser.add_argument('--loss-goal', default=0, type=float, help='average loss goal to stop training at')
parser.add_argument('--augmentation-multiplicity', '--am', default=1, type=int, help='Use multiple augmentations per batch, does not increate batch size')
parser.add_argument('--weight-standardization', '--ws', action='store_true', help='Replace Conv2D layers that come before normallization layers with weight standardization version')
parser.add_argument('--cosine-anealing', '--ca', action='store_true', help='use learning rate cosine anealing scheduler')
parser.add_argument('--eta-min', default=0, type=float, help='min learning rate when cosine anealing')
parser.add_argument('--exponential-moving-average', '--ema', action='store_true', help='use exponential moving average scheduler')
parser.add_argument('--ema-decay', default=0.9, type=float, help='decay for exponential moving average')
parser.add_argument('--weight-decay', default=0, type=float, help='weight decay used in optimizer')
parser.add_argument('--num-workers', default=8, type=int, help='number of workers (dataset download)')
parser.add_argument('--checkpoint-every', default=10, type=int, help='number of epochs to checkpoint the model')
parser.add_argument('--accumulation-steps', default=1, type=int, help='number of steps to accumulate gradients over')

parser.add_argument('--unlearn', '-u', action='store_true', help='wether to unlearn the model')
parser.add_argument('--forgetset-size', '--fs', default=10000, type=int, help='chooses a random forget set  of this size')
parser.add_argument('--load-after-peft', action='store_true', help='wether to load pretrained weights after peft')

parser.add_argument('--differential-privacy', '--dp', default=None, type=str, choices=('opacus', 'fast-dp'), help='wether to train the model with differential privacy')
parser.add_argument('--max-physical-batch-size', '--maxbs', default=128, type=int, help='maximum physical batch size for holding grad samples in memory')
parser.add_argument('--epsilon', default=8.0, type=float, help='epsilon for differential privacy')
parser.add_argument('--delta', default=1e-5, type=float, help='delta for differential privacy')
parser.add_argument('--max-grad-norm', default=1.0, type=float, help='maximum gradient norm for differential privacy')
parser.add_argument('--grad-sample-mode', default='hooks', type=str, help='opacus mode for computing per sample gradients, no-op uses functorch')
parser.add_argument('--no-fix-dp', action='store_true', help='do not fix the model to allow differential privacy')
parser.add_argument('--fix-dp-kwargs', default={}, nargs='*', action=ParseKwargs, help='kwargs to use in fixing the model for differential privacy')

parser.add_argument('--peft', default=None, type=str, help='the peft method to use, either lora, prune or prune-grads',
                    choices=('lora', 'prune-features', 'prune-weights', 'conv-adapter', 'linear-adapter', 'freeze'))
parser.add_argument('--peft-targets', nargs='*', type=str, help='list of target modules to apply peft on')
parser.add_argument('--lora-alpha', default=32, type=int, help='alpha for LoRA')
parser.add_argument('--lora-dropout', default=1e-1, type=float, help='dropout for LoRA')
parser.add_argument('--peft-ratio', default=[4], nargs='*', type=int, help='The ratio to decrease the features width in adapter peft')
parser.add_argument('--peft-modules', default=[], nargs='*', type=str, help='modules from torch.nn to apply peft on')
parser.add_argument('--global-pruning', action='store_true', help='should pruning be global for all layers')
parser.add_argument('--pruning-importance', default='GroupNormImportance', type=str, help='importance function to use in pruning')
parser.add_argument('--prune-grads', action='store_true', help='accumulate grads for epoch before pruning')