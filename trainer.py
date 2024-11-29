import torch
import time
import tester
import static


# Train model
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, weights_path, epochs=200, accumulation_steps=1, checkpoint_model=10):
    training_start_time = time.time()

    for epoch in range(1, epochs + 1):
        # Train for one epoch and calculate the average loss
        start_time = time.time()
        epoch_loss = train_epoch(model, train_loader, criterion, optimizer, accumulation_steps)
        end_time = time.time()
        print(f"Epoch {epoch} finished with loss {epoch_loss} in {end_time - start_time} seconds")

        # Perform scheduler step
        scheduler.step()
        
        # Checkpoint model
        if epoch % checkpoint_model == 0:
            # Save model weights
            torch.save(model.state_dict(), weights_path + '.checkpoint')

            # Output model statistics
            test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion)
            print(f"Checkpoint model at epoch {epoch} with: \n" +
                  f"Test loss: {test_avg_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    # Save model weights
    torch.save(model.state_dict(), weights_path)

    # Output model statistics
    test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion)
    training_end_time = time.time()
    print(f"Training finished in {training_end_time - training_start_time} seconds: \n" +
          f"Test loss: {test_avg_loss:.4f}, Test accuracy: {test_accuracy:.4f}")


# Train model for one epoch
def train_epoch(model, train_loader, criterion, optimizer, accumulation_steps):
    running_loss = 0.0
    model.train() # Set the model to training mode

    for batch_index, (inputs, labels) in enumerate(train_loader):
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.DEVICE), labels.to(static.DEVICE)
        
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