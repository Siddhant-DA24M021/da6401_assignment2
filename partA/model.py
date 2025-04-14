import torch
import torch.nn as nn

class CNNModel(nn.Module):
  def __init__(self, image_size, in_channels=3, num_classes=10,
                num_filters=[64, 64, 64, 64, 64], kernel_size=[3, 3, 3, 3, 3],
                activation_fn=nn.ReLU, fc_layer_size=2048,
                batchnorm=False, dropout=0.0):

    super().__init__()

    h, w = image_size

    # Block 1
    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=kernel_size[0])
    h, w = h - kernel_size[0] + 1, w - kernel_size[0] + 1
    self.batchnorm1 = nn.BatchNorm2d(num_filters[0]) if batchnorm else nn.Identity()
    self.activation1 = activation_fn()
    self.dropout1 = nn.Dropout2d(dropout) if dropout!=0 else nn.Identity()
    self.maxpool1 = nn.MaxPool2d(2, 2)
    h, w = h//2, w//2

    # Block 2
    self.conv2 = nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size[1])
    h, w = h - kernel_size[1] + 1, w - kernel_size[1] + 1
    self.batchnorm2 = nn.BatchNorm2d(num_filters[1]) if batchnorm else nn.Identity()
    self.activation2 = activation_fn()
    self.dropout2 = nn.Dropout2d(dropout) if dropout!=0 else nn.Identity()
    self.maxpool2 = nn.MaxPool2d(2, 2)
    h, w = h//2, w//2

    # Block 3
    self.conv3 = nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size[2])
    h, w = h - kernel_size[2] + 1, w - kernel_size[2] + 1
    self.batchnorm3 = nn.BatchNorm2d(num_filters[2]) if batchnorm else nn.Identity()
    self.activation3 = activation_fn()
    self.dropout3 = nn.Dropout2d(dropout) if dropout!=0 else nn.Identity()
    self.maxpool3 = nn.MaxPool2d(2, 2)
    h, w = h//2, w//2

    # Block 4
    self.conv4 = nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size[3])
    h, w = h - kernel_size[3] + 1, w - kernel_size[3] + 1
    self.batchnorm4 = nn.BatchNorm2d(num_filters[3]) if batchnorm else nn.Identity()
    self.activation4 = activation_fn()
    self.dropout4 = nn.Dropout2d(dropout) if dropout!=0 else nn.Identity()
    self.maxpool4 = nn.MaxPool2d(2, 2)
    h, w = h//2, w//2

    # Block 5
    self.conv5 = nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size[4])
    h, w = h - kernel_size[4] + 1, w - kernel_size[4] + 1
    self.batchnorm5 = nn.BatchNorm2d(num_filters[4]) if batchnorm else nn.Identity()
    self.activation5 = activation_fn()
    self.dropout5 = nn.Dropout2d(dropout) if dropout!=0 else nn.Identity()
    self.maxpool5 = nn.MaxPool2d(2, 2)
    h, w = h//2, w//2

    # Flattening layer
    self.flatten = nn.Flatten()

    # Fully connected layer
    self.fc_layer = nn.Linear(in_features=num_filters[4] * h * w, out_features=fc_layer_size)
    self.batchnorm_fc = nn.BatchNorm1d(fc_layer_size) if batchnorm else nn.Identity()
    self.act_fc = activation_fn()
    self.drop_fc = nn.Dropout(dropout) if dropout!=0 else nn.Identity()

    # Output layer
    self.out = nn.Linear(in_features=fc_layer_size, out_features=num_classes)

  def forward(self, x):
    # Block 1
    x = self.conv1(x)
    x = self.batchnorm1(x)
    x = self.activation1(x)
    x = self.dropout1(x)
    x = self.maxpool1(x)

    # Block 2
    x = self.conv2(x)
    x = self.batchnorm2(x)
    x = self.activation2(x)
    x = self.dropout2(x)
    x = self.maxpool2(x)

    # Block 3
    x = self.conv3(x)
    x = self.batchnorm3(x)
    x = self.activation3(x)
    x = self.dropout3(x)
    x = self.maxpool3(x)

    # Block 4
    x = self.conv4(x)
    x = self.batchnorm4(x)
    x = self.activation4(x)
    x = self.dropout4(x)
    x = self.maxpool4(x)

    # Block 5
    x = self.conv5(x)
    x = self.batchnorm5(x)
    x = self.activation5(x)
    x = self.dropout5(x)
    x = self.maxpool5(x)

    # Flatten
    x = self.flatten(x)

    # Fully connected layers
    x = self.fc_layer(x)
    x = self.batchnorm_fc(x)
    x = self.act_fc(x)
    x = self.drop_fc(x)

    # Output layer
    x = self.out(x)

    return x