import torch
import os
import tester


# Train model
def train(model_name, model, train_loader, test_loader, criterion, optimizer, scheduler, device, max_epochs=200, accumulation_steps=1, min_delta=1e-5, patience=5, checkpoint_every=5):
    epoch_loss = float('inf')
    stop_counter = 0
    epoch = 0

    os.makedirs(f'checkpoints/{model_name}', exist_ok=True)
    
    for epoch in range(1, max_epochs + 1):
        # Train for one epoch and calculate the average loss
        previous_loss, epoch_loss = epoch_loss, train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps)
        print(f"Epoch {epoch} finished with loss {epoch_loss}")

        # Perform scheduler step
        scheduler.step()

        # Check improvement in loss
        if abs(previous_loss - epoch_loss) < min_delta:
            stop_counter += 1
        else: # Reset counter if there's significant improvement
            stop_counter = 0

        # Stop training if no improvement for 'patience' epochs
        if stop_counter >= patience:
            print(f"Stopping early at epoch {epoch}")
            break
        
        # Checkpoint model
        if epoch % checkpoint_every == 0:
            # Save model weights
            torch.save(model.state_dict(), f"checkpoints/{model_name}/epoch{epoch}.pth")

            # Output model statistics
            test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion, device)
            print(f"Checkpoint model at epoch {epoch} with: \n" +
                  f"Test loss: {test_avg_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    # Save model weights
    torch.save(model.state_dict(), f"weights/{model_name}.pth")

    # Output model statistics
    test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion, device)
    print(f"Training finished after {epoch} epochs: \n" +
          f"Test loss: {test_avg_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


# Train model for one epoch
def train_epoch(model, train_loader, criterion, optimizer, device, accumulation_steps):
    running_loss = 0.0
    model.train() # Set the model to training mode

    for batch_index, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()

        # Compute predictions
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        if (batch_index + 1) % accumulation_steps == 0:
            optimizer.step()

    # Return average loss for the epoch
    return running_loss / len(train_loader)