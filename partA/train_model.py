import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils import data_transformations, get_train_and_val_dataloaders, get_test_dataloader
from model import CNNModel
from training_logic import train_model
from sweep_utils import get_activation_function, get_kernel_size, get_num_filters


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set up data directory path
TRAIN_DATA_DIR = "./inaturalist_12K/train"
TEST_DATA_DIR = "./inaturalist_12K/val"

def main():

    # Image resize size
    image_size = (224, 224)

    # Get the trainloader with complete train dataset (No validation set)
    trainloader, _, classnames = get_train_and_val_dataloaders(TRAIN_DATA_DIR, image_size=image_size, valset_size=0, data_augment=True, batch_size=16)

    # Define the model and move to the device
    best_model = CNNModel(image_size, num_filters=get_num_filters("equal32"), kernel_size=get_kernel_size("decreasing"),
                        activation_fn=get_activation_function("mish"), batchnorm=True, dropout=0, fc_layer_size=2048)
    best_model.to(device)

    # Define the criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.parameters(), lr=0.0001)


    epochs = 10

    # Training Loops
    for epoch in range(epochs):

        # Set the model in train mode
        best_model.train()

        # Metrics to keep track of
        running_loss = 0
        correct = 0
        total = 0

        for data in tqdm(trainloader):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = best_model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Metric update
            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

        train_epoch_loss = running_loss / total
        train_epoch_accuracy = 100 * correct / total

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_epoch_loss:.3f}, Train Acc: {train_epoch_accuracy:.2f}%")

    # Best model evaluation
    # Load the test dataset
    testloader = get_test_dataloader(TEST_DATA_DIR, image_size=image_size, batch_size=16)

    # Set the model in evaluation mode
    best_model.eval()

    # Values to keep track of
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader):

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

    test_loss = running_loss / total
    test_accuracy = (correct / total) * 100

    print(f"\nTest Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.2f}%")



if __name__=="__main__":
    main()