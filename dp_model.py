import torch
import trainer
import tester
import loader
import static

from torch import nn, optim
from fastDP import PrivacyEngine

def main():
    train_loader, test_loader = loader.load_dataset(static.DATASET_NAME, static.TRANSFORM, static.TRANSFORM)
    model = loader.load_model(static.BASE_MODEL, static.DEVICE, weights_from='weights/dp_model.pth')
    criterion =  nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=static.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=static.MAX_EPOCHS)

    privacy_engine = PrivacyEngine(
        model,
        batch_size=static.BATCH_SIZE,
        sample_size=static.DATASET_SIZE,
        epochs=static.MAX_EPOCHS,
        target_epsilon=static.EPSILON,
        target_delta=static.DELTA,
        max_grad_norm=static.MAX_GRAD_NORM
    )

    privacy_engine.attach(optimizer)
    trainer.train('dp_model', model, train_loader, test_loader, criterion, optimizer, scheduler, static.DEVICE, max_epochs=static.MAX_EPOCHS, 
                  min_delta=static.MIN_DELTA, patience=static.PATIENCE, checkpoint_every=static.CHECKPOINT_EVERY)

if __name__ == '__main__':
    main()