import torch
import time
import tester
import static
import itertools

from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.func import grad_and_value, vmap, functional_call
from opacus.grad_sample.functorch import make_functional

# Train model
def train(model, train_loader, test_loader, criterion, optimizer, weights_path, schedulers=[], epochs=200, checkpoint_every=10, state_dict={}, accumulation_steps=1,
          loss_goal=0, differential_privacy=None, ma_model=None, max_physical_batch_size=128, augmentation_multiplicity=1, grad_sample_mode='no_op', forget_loader=None):
    training_start_time = time.time()
    state_dict['epochs'] = []
    state_dict['checkpoints'] = []
    checkpoint_model = ma_model.module if ma_model else model

    for epoch in range(1, epochs + 1):
        # Train for one epoch and calculate the average loss
        start_time = time.time()
        match differential_privacy:
            case 'opacus':
                with BatchMemoryManager(
                    data_loader=train_loader, 
                    max_physical_batch_size=max_physical_batch_size, 
                    optimizer=optimizer
                ) as memory_safe_train_loader:
                    if grad_sample_mode == 'no_op':
                        epoch_loss, epoch_accuracy = train_epoch_dp_functorch(model, memory_safe_train_loader, criterion, optimizer, augmentation_multiplicity)
                    else:
                        epoch_loss, epoch_accuracy = train_epoch_dp(model, memory_safe_train_loader, criterion, optimizer, augmentation_multiplicity)
            case 'fast-dp':
                epoch_loss, epoch_accuracy = train_epoch(model, train_loader, criterion, optimizer, accumulation_steps=accumulation_steps)
            case _:
                if forget_loader:
                    epoch_loss, forget_epoch_loss, epoch_accuracy, forget_epoch_accuracy = neg_grad(model, train_loader, forget_loader, criterion, optimizer)
                else:
                    epoch_loss, epoch_accuracy = train_epoch(model, train_loader, criterion, optimizer, accumulation_steps=accumulation_steps)

        end_time = time.time()
        print(f"Epoch {epoch} - Train loss: {epoch_loss}, Train accuracy: {epoch_accuracy} , Time: {(end_time - start_time):.2f}s")

        epoch_state = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'time': end_time - start_time
        }

        if forget_loader:
            print(f"Epoch {epoch} - Forget loss: {forget_epoch_loss}, Forget accuracy: {forget_epoch_accuracy} , Time: {(end_time - start_time):.2f}s")

            epoch_state |= {
                'forget_loss': forget_epoch_loss, 
                'forget_accuracy': forget_epoch_accuracy
            }
        
        state_dict['epochs'].append(epoch_state)

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
            test_loss, test_accuracy = tester.test(checkpoint_model, test_loader, criterion)
            print(f"Checkpoint model at epoch {epoch} with: \n" +
                  f"Test loss: {test_loss}, Test accuracy: {test_accuracy:.2f}")

            state_dict['checkpoints'].append({
                'epoch': epoch,
                'loss': test_loss,
                'accuracy': test_accuracy,
            })

            # Save model weights
            torch.save(state_dict | {'model': checkpoint_model.state_dict()}, weights_path + '.checkpoint') # Merge state dict so it doesn't sit on memory

    # Output model statistics
    test_avg_loss, test_accuracy = tester.test(checkpoint_model, test_loader, criterion)
    training_end_time = time.time()
    print(f"Training finished in {training_end_time - training_start_time} seconds: \n" +
          f"Test loss: {test_avg_loss}, Test accuracy: {test_accuracy:.2f}")
    
    state_dict['time'] = training_end_time - training_start_time
    state_dict['loss'] = test_avg_loss
    state_dict['accuracy'] = test_accuracy
    state_dict['model'] = checkpoint_model.state_dict()
    
    # Save model weightsmodel
    torch.save(state_dict, weights_path)


# Train model for one epoch
def train_epoch(model, train_loader, criterion, optimizer, keep_gradients=False, accumulation_steps=1):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train() # Set the model to training mode

    for idx, (inputs, labels) in enumerate(train_loader):
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
        if not keep_gradients and ((idx + 1) % accumulation_steps == 0 or idx == len(train_loader) - 1) :
            optimizer.step()
            optimizer.zero_grad()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy

# Train model for one epoch with differntial privacy augmentations
def train_epoch_dp(model, train_loader, criterion, optimizer, augmentation_multiplicity):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train() # Set the model to training mode

    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.CUDA), labels.to(static.CUDA)

        # Cut incomplete augmentations
        augmentation_remainder = inputs.size(0) % augmentation_multiplicity

        if augmentation_remainder != 0:
            inputs = inputs[:-augmentation_remainder,...]
            labels = labels[:-augmentation_remainder]

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
        if augmentation_multiplicity != 0:
            for param in model.parameters():
                param.grad_sample = torch.mean(torch.stack(torch.split(param.grad_sample, augmentation_multiplicity)), dim=1)
        
        # Adjust learning weights and zero gradients
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / (len(train_loader) * augmentation_multiplicity)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy

# Train model for one epoch with differntial privacy augmentations
def train_epoch_dp_functorch(model, train_loader, criterion, optimizer, augmentation_multiplicity):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.train() # Set the model to training mode

    fmodel, _fparams = make_functional(model)

    def compute_sample_loss(params, input, label):
        inputs, labels = input.unsqueeze(0), label.unsqueeze(0)
        outputs = fmodel(params, inputs)
        loss = criterion(outputs, labels)

        return loss

    params = list(model.parameters())

    compute_grad = grad_and_value(compute_sample_loss) # Returns loss and gradients
    compute_grad_samples = vmap(compute_grad, in_dims=(None, 0, 0)) # compute grads over groups of batches

    for inputs, labels in train_loader:
        # Move inputs and labels to the specified device
        inputs, labels = inputs.to(static.CUDA), labels.to(static.CUDA)

        with torch.no_grad():
            # Compute predictions
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            # Update the running total of correct predictions and samples
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        # Compute the loss and its gradients
        grad_samples, grad_losses = compute_grad_samples(params, inputs, labels)
        grad_samples = [grad.detach() for grad in grad_samples]
        loss = torch.mean(grad_losses)
        running_loss += loss.item()

        # Average grad samples over augmentations
        for param, grad in zip(params, grad_samples):
            param.grad_sample = torch.mean(torch.stack(torch.split(grad, augmentation_multiplicity)), dim=1)

        # Adjust learning weights and zero gradients
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / (len(train_loader) * augmentation_multiplicity)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy

# NegGrad+ one epoch
def neg_grad(model, retain_loader, forget_loader, criterion, optimizer):
    retain_running_loss, forget_running_loss = 0.0, 0.0
    retain_correct_predictions, forget_correct_predictions = 0, 0
    retain_total_predictions, forget_total_predictions = 0, 0
    model.train() # Set the model to training mode

    loader = zip(retain_loader, itertools.cycle(forget_loader))

    for (retain_inputs, retain_labels), (forget_inputs, forget_labels) in zip(retain_loader, forget_loader):
        # Move inputs and labels to the specified device
        retain_inputs, retain_labels = retain_inputs.to(static.CUDA), retain_labels.to(static.CUDA)
        forget_inputs, forget_labels = forget_inputs.to(static.CUDA), forget_labels.to(static.CUDA)

        # Compute predictions
        retain_outputs = model(retain_inputs)
        forget_outputs = model(forget_inputs)
        _, retain_predictions = torch.max(retain_outputs.data, 1)
        _, forget_predictions = torch.max(forget_outputs.data, 1)

        # Update the running total of correct predictions and samples
        retain_correct_predictions += (retain_predictions == retain_labels).sum().item()
        forget_correct_predictions += (forget_predictions == forget_labels).sum().item()
        retain_total_predictions += retain_labels.size(0)
        forget_total_predictions += forget_labels.size(0)

        # Compute the loss and its gradients
        retain_loss = criterion(retain_outputs, retain_labels)
        forget_loss = criterion(forget_outputs, forget_labels)
        retain_running_loss += retain_loss.item()
        forget_running_loss += forget_loss.item()
        retain_loss.backward()
        (-1.0 * forget_loss).backward()

        # Adjust learning weights and zero gradients
        optimizer.step()
        optimizer.zero_grad()

    # Calculate the average loss and accuracy
    retain_avg_loss = retain_running_loss / len(retain_loader)
    forget_avg_loss = forget_running_loss / len(retain_loader)
    retain_accuracy = 100 * retain_correct_predictions / retain_total_predictions
    forget_accuracy = 100 * forget_correct_predictions / forget_total_predictions

    return retain_avg_loss, forget_avg_loss, retain_accuracy, forget_accuracy