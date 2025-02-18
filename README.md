# DP Unlearning
This repository is part of a research on using differential privacy to get a model for better unlearning, this research suggests the following training method:
1. Training the model using diffential privacy to obtain a model with similar train and test accuracy.
2. Fine-tune the differentialy private model on a small amount of parameters, that way storing critical information "remembering" the dataset will only sit on a small amount of parameters.
When we want to unlearn a specific set in the dataset, we unlearn only the specific parameters that were trained in the fine-tuning process, which will hopefully hurt the model less when unlearning.

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
| Normal | X | 5.6M | 88.02 | 99.97 |
| DP | X | 5.6M | 47.42 | 48.90 |
| LoRa (Rank=16) | X | 0.6M | 87.73 | 95.13 |
| LoRa (Rank=8) | DP | 0.3M | 84.91 | 90.66 |
| LoRa (Rank=16) | DP | 0.6M | 87.55 | 95.76 |
| LoRa (Rank=32) | DP | 1.2M | 88.23 | 96.83 |