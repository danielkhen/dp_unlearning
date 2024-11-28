import torch
import trainer
import loader
import fine_tuning
import static

from torch import nn, optim
from torch.nn import Linear, Conv2d

lora_supported_types = (Linear, Conv2d)

def main():
    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, static.TRANSFORM, static.TRANSFORM)
    model = loader.load_model(static.BASE_MODEL, static.DEVICE, weights_from='weights/dp_model.pth')
    target_modules = [name for name, module in model.blocks.named_modules() if isinstance(module, lora_supported_types)]
    
    for module in target_modules:
        fine_tuning.prune_gradients(module, 'weight', 0.9)

    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=static.LR, weight_decay=static.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=static.MAX_EPOCHS)
    
    print(f"trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer.train('dp_lora_ft', model, train_loader, test_loader, criterion, optimizer, scheduler, static.DEVICE, max_epochs=static.MAX_EPOCHS, 
                  min_delta=static.MIN_DELTA, patience=static.PATIENCE, checkpoint_every=static.CHECKPOINT_EVERY)

if __name__ == '__main__':
    main()