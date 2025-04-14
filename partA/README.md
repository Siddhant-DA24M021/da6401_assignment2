# DA6401 Assignment 2 Part-A
# Siddhant Baranwal (DA24M021)


### Set Up
To set up the environment, make sure you have the following Python packages installed:

- `torch` - Core deep learning library
- `torchvision` - Utilities for image processing with PyTorch
- `matplotlib` - Visualization library
- `numpy` - Numerical computing
- `tqdm` - Progress bars
- `ipykernel` - Jupyter kernel support
- `wandb` - Experiment tracking and logging

### Installation
You can install all dependencies using pip:

```bash
pip install torch torchvision matplotlib numpy tqdm ipykernel wandb
```

### Files
1. part_a.ipynb :- Main notebook (where all the magic happened :)
2. data_utils.py :- Transform images and create dataloaders.
3. model.py :- Contains the main model architecture.
4. training_logic.py :- Contains the training logic.
5. sweep_utils.py :- Contains some helper functions for sweeping.
6. hyperparameter_sweep.py :- WandB sweeping code for hyperparameter tuning.
7. best_model.py :- Training and Evaluation of best model parameters.

### Metrics of the best model :-
### Training Loss of best model :- 1.608
### Training Accuracy of best model :- 44.47%
### Test Loss of best model :- 1.756
### Test Accuracy of best model :- 40.50%