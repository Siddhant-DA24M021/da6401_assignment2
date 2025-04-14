import torch
import torch.nn as nn

def get_activation_function(name):

  """This function returns the activation function based on the name passed"""

  if name == "leaky_relu":
    return nn.LeakyReLU
  elif name == "parametric_relu":
    return nn.PReLU
  elif name == "gelu":
    return nn.GELU
  elif name == "silu":
    return nn.SiLU
  elif name == "mish":
    return nn.Mish
  return nn.ReLU



def get_num_filters(name):

  """This function is a helper function for the hyper parameter sweep.
     As we need different filter sizes for the model to perform the sweep."""

  if name == "equal16":
    return [16, 16, 16, 16, 16]
  elif name == "equal32":
    return [32, 32, 32, 32, 32]
  elif name == "equal64":
    return [64, 64, 64, 64, 64]
  elif name == "doubling16":
    return [16, 32, 64, 128, 256]
  elif name == "doubling32":
    return [32, 64, 128, 256, 512]
  elif name == "halving256":
    return [256, 128, 64, 32, 16]
  else:
    return [100, 80, 50, 80, 100]



def get_kernel_size(name):

  """This function is a helper function for the hyper parameter sweep.
     As we need different kernel sizes for the model to perform the sweep."""

  if name == "constant5":
    return [5, 5, 5, 5, 5]
  elif name == "constant7":
    return [7, 7, 7, 7, 7]
  elif name == "decreasing":
    return [5, 5, 3, 3, 1]
  elif name == "increasing":
    return [1, 3, 3, 5, 5]
  return [3, 3, 3, 3, 3]
