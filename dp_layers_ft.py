import torch
import trainer
import loader
import static

from torch import nn, optim



def main():
    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, static.TRANSFORM, static.TRANSFORM)
    model = loader.load_model(static.BASE_MODEL, static.DEVICE, weights_from='weights/dp_model.pth')

    for param in list(model.module.layer3.parameters()) + list(model.module.layer4.parameters()):
        param.requires_grad = False

    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=static.LR, momentum=static.MOMENTUM, weight_decay=static.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=static.MAX_EPOCHS)
    
    print(f"trainable parameters {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer.train('dp_layers_ft', model, train_loader, test_loader, criterion, optimizer, scheduler, static.DEVICE, max_epochs=static.MAX_EPOCHS, 
                  min_delta=static.MIN_DELTA, patience=static.PATIENCE, checkpoint_every=static.CHECKPOINT_EVERY)

if __name__ == '__main__':
    main()