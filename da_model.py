import trainer
import loader
import static

from torch import nn, optim

def main():
    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, static.DA_TRANSFORM, static.TRANSFORM)
    model = loader.load_model(static.BASE_MODEL, static.DEVICE, weights_from='weights/da_model.pth')
    optimizer = optim.AdamW(model.parameters(), lr=static.LR, weight_decay=static.WEIGHT_DECAY)
    criterion =  nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=static.MAX_EPOCHS)

    trainer.train('da_model', model, train_loader, test_loader, criterion, optimizer, scheduler, static.DEVICE, max_epochs=static.MAX_EPOCHS, min_delta=static.MIN_DELTA, patience=static.PATIENCE, checkpoint_every=static.CHECKPOINT_EVERY)

if __name__ == '__main__':
    main()