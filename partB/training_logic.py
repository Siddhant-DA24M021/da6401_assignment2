import torch
from tqdm import tqdm

def train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=10):

  """This function trains and evaluates the model for the specified number of epochs."""

  # Metrics to keep track of
  train_epoch_losses = []
  train_epoch_accuracies = []
  val_epoch_losses = []
  val_epoch_accuracies = []

  for epoch in range(epochs):

    # Set the model in train mode
    model.train()

    # Epoch Metrics
    train_running_loss = 0
    train_correct = 0
    train_total = 0

    for data in tqdm(trainloader):

      inputs, labels = data[0].to(device), data[1].to(device)

      optimizer.zero_grad()

      outputs = model(inputs)

      loss = criterion(outputs, labels)

      loss.backward()
      optimizer.step()

      # Update metrics
      train_running_loss += loss.item() * inputs.size(0)
      train_total += inputs.size(0)
      train_correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

    train_epoch_loss = train_running_loss / train_total
    train_epoch_accuracy = 100 * train_correct / train_total

    train_epoch_losses.append(train_epoch_loss)
    train_epoch_accuracies.append(train_epoch_accuracy)

    # Set the model to evaluation mode
    model.eval()

    # Epoch Metrics
    val_running_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
      for data in tqdm(valloader):

        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Update Metrics
        val_running_loss += loss.item() * inputs.size(0)
        val_total += inputs.size(0)
        val_correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

    val_epoch_loss = val_running_loss / val_total
    val_epoch_accuracy = 100 * val_correct / val_total

    val_epoch_losses.append(val_epoch_loss)
    val_epoch_accuracies.append(val_epoch_accuracy)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_epoch_loss:.3f}, Train Acc: {train_epoch_accuracy:.2f}%, Val Loss: {val_epoch_loss:.3f}, Val Acc: {val_epoch_accuracy:.2f}%")

  return train_epoch_losses, train_epoch_accuracies, val_epoch_losses, val_epoch_accuracies