import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random
from data_utils import data_transformations, get_train_and_val_dataloaders, get_test_dataloader

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set up data directory path
TRAIN_DATA_DIR = "./inaturalist_12K/train"
TEST_DATA_DIR = "./inaturalist_12K/val"


def main():

    # Image transformations used in EfficientNet_V2_S
    image_size = (384, 384)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Get training transformer
    train_transforms = data_transformations(image_size=image_size, mean=mean, std=std, data_augment=True)

    # Get the trainloader and valloader from complete train dataset
    trainloader, _, classnames = get_train_and_val_dataloaders(TRAIN_DATA_DIR, train_transforms, valset_size=0, batch_size=16)


    # Get testing transformer
    test_transforms = data_transformations(image_size=image_size, mean=mean, std=std, data_augment=False)

    # Get the trainloader and valloader from complete train dataset
    testloader = get_test_dataloader(TEST_DATA_DIR, test_transforms, batch_size=16)


    # Load the model
    model = models.efficientnet_v2_s(weights="DEFAULT")

    # Change the output layer
    in_features = model.classifier[1].in_features
    num_classes = 10
    model.classifier[1] = nn.Linear(in_features, num_classes)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last classifier layer
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    # Shift the model to the device being used
    model.to(device)


    # Define the criterion and optimizer to be used in training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Strategy used:- Gradual Unfreezing of layers

    EPOCHS = 10
    train_epoch_losses = []
    train_epoch_accuracies = []
    for epoch in range(EPOCHS):

        # Unfreeze a layer from last after every 2 epochs
        if epoch % 2 == 0:
            for params in model.features[-1 - (epoch//2)].parameters():
                params.requires_grad = True
            
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

    print(f"Train Loss: {train_epoch_losses[-1]:.3f}, Train Acc: {train_epoch_accuracies[-1]:.2f}%")

    # Set the model in evaluation mode
    model.eval()

    # Values to keep track of
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(testloader):

            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)
            correct += torch.sum(labels == torch.argmax(outputs, dim=1)).item()

    test_loss = running_loss / total
    test_accuracy = (correct / total) * 100

    print(f"\nTest Loss: {test_loss:.3f}, Test Acc: {test_accuracy:.2f}%")


if __name__=="__main__":
    main()