# DP Unlearning
This repository is part of a research on using differential privacy to achieve a model that better recovers from unlearning, this research suggests the following training method:
1. Training the model using diffential privacy to obtain a model with similar train and test accuracy.
2. Fine-tune the differentialy private model on a small amount of parameters, that way storing critical information/remembering the dataset in a small amount of parameters.
When we want to unlearn a specific forget set in the dataset, we unlearn only the on specific parameters that were trained in the fine-tuning process, which will hopefully hurt the model's accuracy less when unlearning.

## Stage 1
In the first stage of this research we want to get the highest accuracy model trained with this method.
We have tested 2 models:
- A variant of Resnet18 using group norm instead of batch norm as batch norm isn't allowed in differential privacy, and weight standardization on each convolutional layer that is followed by a normalization layer.
-  ViT-Tiny, a small transformer model with width 4 and depth 12.
The models were all trained on the CIFAR10 dataset, first normally, then with differential privacy, and then fine-tuned on the DP model with different methods.
The following table shows the accuracies achieved with the different methods tried:

### Resnet18
| Method | Parameters | Test accuracy | Train accuracy |
|---|---|---|---|
| Normal | 11.2M | 89.76 | 98.94 |
| DP | 11.2M | 68.13 | 68.91 |
| Conv Adapter | 1.7M | 88.09 | 95.26 |
| LoRa | 1.5M | 85.95 | 92.15 |
| Channel Pruning | 1.4M | 85.73  | 91.93 |

### ViT-Tiny
| Method | Trained on | Parameters | Test accuracy | Train accuracy |
|---|---|---|---|---|
| Normal | X | 5.6M | 88.02 | 99.8 |
| DP | X | 5.6M | 47.42 | 48.90 |
| LoRa (Rank=16) | X | 0.6M | 87.73 | 95.13 |
| LoRa (Rank=8) | DP | 0.3M | 84.91 | 90.66 |
| LoRa (Rank=16) | DP | 0.6M | 87.55 | 95.76 |
| LoRa (Rank=32) | DP | 1.2M | 88.23 | 96.83 |


## Stage 2
Test unlearning (until test and forget set accuracies are the same) the original model against the new method, compare both test accuracies, hopefully achieving higher accuracy on the new method.

### ViT-Base
#### Notes
- The pretrained model refers to ViT-Base with input size 224, trained on ImageNet, thus in the following training Cifar10 images were resized from size 32 to 224.
- Each model was trained for 5 epochs which means the models are not fully converged but close.
- Train accuracies are not exact as they were captured mid training while the models weights were still changing.

#### Results

| Method | Trained on | Parameters | Test accuracy | Train accuracy | Comment |
|---|---|---|---|---|---|
| Normal | pretrained | 85.8M | 99.02 | 99.9 | |
| DP (eps=8) | pretrained | 85.8M | 97.8 | 98.45 | |
| DP (eps=3) | pretrained | 85.8M | 97.57 | 98.22 | |
| DP (eps=2) | pretrained | 85.8M | 97.5 | 98.14 | |
| DP (eps=1) | pretrained | 85.8M | 97.32 | 97.85 | |
| LoRa (Rank=2-64) | DP (eps=1) and pretrained | 1M - 10M | 98.7 - 98.85 | | Power of 2 rank from 2 to 64 LoRa on attention and MLP linear layers |
| LoRa (Rank=4, 1) | DP (eps=1) | 0.65M | 98.77 | 99.7 | rank 4 LoRa on attention and MLP linear layers and rank 1 out of 3 on convolutional patch embedding |

#### Conclusions
- LoRa failed to achieve higher accuracies for higher ranks, also choosing the DP model as a starting point didn't move this barrier.
- The use of pretrained models allowed high accuracy for both LoRa with low ranks and differential privacy with low epsilons.
- The accuracy of the model is barely hurt with less then 1% of the original parameters.

#### Unlearning
Both the Normal model and the LoRa (Rank=4, 1) model were unlearned using the NegGrad+ algorithm (descent on the retain and ascent on the forget) until the accuracy on the forget set equaled the accuracy on the test set. Remarkably the LoRa model ended up with a test set accuracy of 98.21, notably higher then the test set accuracy on the normal model of 97.66
