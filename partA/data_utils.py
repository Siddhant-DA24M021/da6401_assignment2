import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

def data_transformations(image_size=(224, 224), data_augment=False):

  """This function returns data transformations for the images data."""

  # Define transformations to be applied (Base Transformations)
  transformations = [
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # NOTE:- I am planning to use EfficientNetV2 for part B of the assignment so using the same values as used for that network
  ]

  # If Augmentation is needed, add them to transform list
  if data_augment:
    transformations += [
      transforms.RandomHorizontalFlip(0.05),
      transforms.RandomVerticalFlip(0.05),
      transforms.RandomRotation(degrees=20),
      transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)], p=0.05),
      transforms.RandomApply([transforms.GaussianBlur(3)], p=0.05)
    ]

  transformer = transforms.Compose(transformations)

  return transformer


def get_train_and_val_dataloaders(train_data_dir, image_size=(224, 224), data_augment=False, valset_size=0.2, batch_size=16):

  """This function returns the dataloader for trainset and validation set and classnames"""

  transformer = data_transformations(image_size, data_augment)

  # Load the total_train dataset
  total_trainset = torchvision.datasets.ImageFolder(root = train_data_dir, transform=transformer)

  # Get the classnames
  classnames = total_trainset.classes

  # Split the total_train data into train data and val data
  labels = [label for _, label in total_trainset.samples]

  if valset_size != 0:
    train_indices, val_indices = train_test_split(
                                    range(len(total_trainset)),
                                    test_size=valset_size,
                                    stratify=labels,
                                    random_state=42
                                    )
  else:
    train_indices = range(len(total_trainset))
    val_indices = []

  # Create the trainset and valset
  trainset = torch.utils.data.Subset(total_trainset, train_indices)
  valset = torch.utils.data.Subset(total_trainset, val_indices)

  # Create the dataloaders
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

  valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

  return trainloader, valloader, classnames



def get_test_dataloader(test_data_dir, image_size=(224, 224), batch_size=16):

  """This function returns the test dataloader"""

  transformer = data_transformations(image_size, False)

  #Download the test data
  testset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transformer)



  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

  return testloader