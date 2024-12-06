import torch
import time
import tester
import static

from loader import MultiTransformDataset
from opacus.utils.batch_memory_manager import BatchMemoryManager
from tqdm import tqdm

# Train model
def train(model, train_loader, test_loader, criterion, optimizer, weights_path, schedulers=[], epochs=200, checkpoint_every=10, state_dict={}, loss_goal=0, differential_privacy=True, ma_model=None, max_physical_batch_size=512):
    training_start_time = time.time()
    state_dict['epochs'] = []
    state_dict['checkpoints'] = []
    checkpoint_model = ma_model.module if ma_model else model

    for epoch in tqdm(range(1, epochs + 1)):
        # Train for one epoch and calculate the average loss
        start_time = time.time()

        if differential_privacy:
            with BatchMemoryManager(
                data_loader=train_loader, 
                max_physical_batch_size=max_physical_batch_size, 
                optimizer=optimizer
            ) as memory_safe_train_loader:
                epoch_loss, epoch_accuracy = train_epoch_dp(model, memory_safe_train_loader, criterion, optimizer)
        else:
            epoch_loss, epoch_accuracy = train_epoch(model, train_loader, criterion, optimizer)

        end_time = time.time()
        print(f"Epoch {epoch} - Train loss: {epoch_loss}, Train accuracy: {epoch_accuracy} , Time: {(end_time - start_time):.2f}s")

        state_dict['epochs'].append({
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'time': end_time - start_time,
        })

        # Stop training if loss achieved goal
        if epoch_loss <= loss_goal:
            print("Training stopped after achieving loss goal")
            break

        # Perform schedulers steps
        for scheduler in schedulers:
            scheduler.step()

        if ma_model:
            ma_model.update_parameters(model)
        
        # Checkpoint model
        if epoch % checkpoint_every == 0:
            # Output model statistics
            test_avg_loss, test_accuracy = tester.test(checkpoint_model, test_loader, criterion)
            print(f"Checkpoint model at epoch {epoch} with: \n" +
                  f"Test loss: {test_avg_loss}, Test accuracy: {test_accuracy:.2f}")

            state_dict['checkpoints'].append({
                'epoch': epoch,
                'loss': test_avg_loss,
                'accuracy': test_accuracy,
            })

            # Save model weights
            torch.save(state_dict | {'model': checkpoint_model.state_dict()}, weights_path + '.checkpoint') # Merge state dict so it doesn't sit on memory

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

    for inputs, labels in tqdm(train_loader):
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.CUDA), labels.to(static.CUDA)

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

        # Adjust learning weights and zero gradients
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy

# Train model for one epoch with differntial privacy augmentations
def train_epoch_dp(model, train_loader, criterion, optimizer):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    augmentation_multiplicity = train_loader.dataset.augmentation_multiplicity if isinstance(train_loader.dataset, MultiTransformDataset) else 1
    model.train() # Set the model to training mode

    for inputs, labels in tqdm(train_loader):
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.CUDA), labels.to(static.CUDA)

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

        # Average grad samples over augmentations
        if augmentation_multiplicity != 1:
            for param in model.parameters():
                param.grad_sample = torch.mean(torch.stack(torch.split(param.grad_sample, augmentation_multiplicity)), dim=0)
        
        # Adjust learning weights and zero gradients
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / (len(train_loader) * augmentation_multiplicity)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy