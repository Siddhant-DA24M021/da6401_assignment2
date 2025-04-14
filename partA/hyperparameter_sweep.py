import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
import random
import os

from data_utils import data_transformations, get_train_and_val_dataloaders, get_test_dataloader
from model import CNNModel
from training_logic import train_model
from sweep_utils import get_activation_function, get_kernel_size, get_num_filters


device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set up data directory path
TRAIN_DATA_DIR = "./inaturalist_12K/train"

def sweep_hyperparameters(config=None):

  with wandb.init(config=config):

    config = wandb.config
    wandb.run.name = f"activation_{str(config.activation)}_filters_{str(config.num_filters)}_lr_{config.learning_rate}_kernel_{config.kernel_size}_fc_size_{config.fc_layer_size}"

    # Log in my details
    wandb.config.update({"NAME": "SIDDHANT BARANWAL", "ROLL NO.": "DA24M021"})

    image_size=(224, 224)

    trainloader, valloader, classnames = get_train_and_val_dataloaders(TRAIN_DATA_DIR, image_size=image_size, data_augment=config.data_augment, valset_size=0.2, batch_size=config.batch_size)


    model = CNNModel(image_size, num_filters=get_num_filters(config.num_filters), kernel_size=get_kernel_size(config.kernel_size),
                     activation_fn=get_activation_function(config.activation), batchnorm=config.batch_norm, dropout=config.dropout,
                     fc_layer_size=config.fc_layer_size)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    train_losses, train_accuracies, val_losses, val_accuracies = train_model(model, trainloader, valloader, criterion, optimizer, device, epochs=10)

    # Log the evaluation metrics

    for epoch in range(len(train_losses)):
      wandb.log({
          "epoch": epoch,
          "train_loss": train_losses[epoch],
          "train_accuracy": train_accuracies[epoch],
          "validation_loss": val_losses[epoch],
          "validation_accuracy": val_accuracies[epoch]
      })

    wandb.log({
        "val_accuracy": val_accuracies[-1]
    })
    


if __name__=="__main__":

    sweep_config = {
        "method" : "bayes",
        "metric" : {"name": "val_accuracy", "goal": "maximize"},
        "parameters" : {
            "data_augment" : {"values" : [True, False]},
            "batch_norm" : {"values" : [True, False]},
            "dropout" : {"values" : [0.0, 0.2, 0.4]},
            "learning_rate" : {"values" : [0.01, 0.001, 0.0005, 0.0001]},
            "activation" : {"values" : ["relu", "leaky_relu", "parametric_relu",
                                        "gelu", "silu", "mish"]},
            "num_filters" : {"values" : ["equal16", "equal32", "equal64", "doubling16", "doubling32", "halving256"]},
            "kernel_size" : {"values" : ["constant3", "constant5", "constant7", "decreasing", "increasing"]},
            "fc_layer_size" : {"values": [2048, 1024, 512]},
            "batch_size": {"values": [8, 16, 32]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project = "da24m021_da6401_assignment2")
    wandb.agent(sweep_id, function = sweep_hyperparameters, count = 50)