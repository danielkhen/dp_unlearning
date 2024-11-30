import torch
import time
import tester
import static


# Train model
def train(model, train_loader, test_loader, criterion, optimizer, scheduler, weights_path, epochs=200, checkpoint_model=10, state_dict={}, dp_transforms=None, use_scheduler=True):
    training_start_time = time.time()
    state_dict['epochs'] = []
    state_dict['checkpoints'] = []

    for epoch in range(epochs):
        # Train for one epoch and calculate the average loss
        start_time = time.time()
        epoch_loss, epoch_accuracy = train_epoch_dp(model, train_loader, criterion, optimizer, dp_transforms) if dp_transforms else train_epoch(model, train_loader, criterion, optimizer)
        end_time = time.time()
        print(f"Epoch {epoch} - Train loss: {epoch_loss}, Train accuracy: {epoch_accuracy} , Time: {(end_time - start_time):.2f}s")

        state_dict['epochs'].append({
            'loss': epoch_loss,
            'time': end_time - start_time,
        })

        # Perform scheduler step
        if use_scheduler:
            scheduler.step()
        
        # Checkpoint model
        if (epoch + 1) % checkpoint_model == 0:
            # Output model statistics
            test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion)
            print(f"Checkpoint model at epoch {epoch} with: \n" +
                  f"Test loss: {test_avg_loss}, Test accuracy: {test_accuracy:.2f}")

            state_dict['checkpoints'].append({
                'epoch': epoch,
                'loss': test_avg_loss,
                'accuracy': test_accuracy,
            })

            state_dict['model'] = model.state_dict()

            # Save model weights
            torch.save(state_dict, weights_path + '.checkpoint')

    # Output model statistics
    test_avg_loss, test_accuracy = tester.test(model, test_loader, criterion)
    training_end_time = time.time()
    print(f"Training finished in {training_end_time - training_start_time} seconds: \n" +
          f"Test loss: {test_avg_loss}, Test accuracy: {test_accuracy:.2f}")
    
    state_dict['time'] = training_end_time - training_start_time
    state_dict['loss'] = test_avg_loss
    state_dict['accuracy'] = test_accuracy
    state_dict['model'] = model.state_dict()
    
    # Save model weights
    torch.save(state_dict, weights_path)



# Train model for one epoch
def train_epoch(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train() # Set the model to training mode

    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.DEVICE), labels.to(static.DEVICE)

        # Zero gradients
        optimizer.zero_grad()

        # Compute predictions
        outputs = model(inputs)
        _, predictions = torch.max(outputs.data, 1)

        # Update the running total of correct predictions and samples
        correct_predictions += (predictions == labels).sum().item()
        total_predictions += labels.size(0)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        running_loss += loss.item()
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy

# Train model for one epoch with differntial privacy augmentations
def train_epoch_dp(model, train_loader, criterion, optimizer, transforms):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train() # Set the model to training mode

    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.DEVICE), labels.to(static.DEVICE)

        # Initialize per gradient sample sum
        for param in model.parameters():
            param.grad_sample_sum = torch.zeros((inputs.size()[0], ) + param.size(), device=static.DEVICE) # Expand in batch size
        
        for transform in transforms:
            # Zero gradients
            optimizer.zero_grad()

            # Apply transform
            augmented_inputs = transform(inputs)

            # Compute predictions
            outputs = model(augmented_inputs)
            _, predictions = torch.max(outputs.data, 1)

            # Update the running total of correct predictions and samples
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Compute the loss and its gradients
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()

            param.grad_sample_sum += param.grad_sample

        # Average the gradients over all augmentations
        for param in model.parameters():
            param.grad_sample = param.grad_sample_sum / len(transforms)
        
        # Adjust learning weights, optimizer should clip already averaged grad samples
        optimizer.step()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / (len(train_loader) * len(transforms))
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy