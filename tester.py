import torch
import static

def test(model, test_loader, criterion):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    model.eval() # Set the model to eval mode

    with torch.no_grad():
        for inputs, labels in test_loader:
            # Move inputs and labels to the specified device
            inputs, labels = inputs.to(static.DEVICE), labels.to(static.DEVICE)
            
            # Compute predictions
            outputs = model(inputs)
            _, predictions = torch.max(outputs.data, 1)

            # Update the running total of correct predictions and samples
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

            # Compute the loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    # Calculate the average loss and accuracy
    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct_predictions / total_predictions

    return avg_loss, accuracy
