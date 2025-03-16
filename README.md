README - Feedforward Neural Network on Fashion-MNIST

**Problem Statement**

In this assignment, we implement a feedforward neural network from scratch and write the backpropagation code for training the network. We use NumPy for matrix and vector operations without any automatic differentiation libraries. The network is trained and tested on the Fashion-MNIST dataset to classify images into 10 categories.

Dataset

The dataset consists of 60,000 training images and 10,000 test images, each of size 28x28 pixels in grayscale. The 10 clothing categories are:

T-shirt/top

Trouser

Pullover

Dress

Coat

Sandal

Shirt

Sneaker

Bag

Ankle boot

Code Structure

The implementation consists of the following Python scripts:

train.py: Implements the feedforward neural network, backpropagation, and training process.

utils.py: Contains helper functions for data preprocessing, weight initialization, and activation functions.

sweep.py: Defines the hyperparameter tuning strategy using Weights & Biases (WandB).

Implementation Details

1. Data Preprocessing

Loaded the Fashion-MNIST dataset using keras.datasets.fashion_mnist.

Normalized the pixel values for efficient training.

Converted labels to one-hot encoding.

Split data into training, validation, and test sets.

2. Feedforward Neural Network

Implemented a flexible architecture that allows customization of:

Number of hidden layers.

Number of neurons per layer.

Activation functions (ReLU, Sigmoid, Tanh for hidden layers, Softmax for output layer).

Loss functions (Cross-Entropy Loss, Mean Squared Error).

Forward propagation implemented in Forward_Propagation().

3. Backpropagation & Optimization Algorithms

Implemented backpropagation in Back_Propagation() to update weights.

Supported multiple optimization algorithms:

Stochastic Gradient Descent (SGD) (Stochastic_GD())

Momentum-based GD (Momentum_GD())

Nesterov Accelerated GD (Nesterov_GD())

RMSProp (RMS_Opt())

Adam (Adam_Opt())

Nadam (NAdam_Opt())

Code is designed for easy extensibility to add new optimization techniques.

4. Hyperparameter Tuning using WandB

Conducted hyperparameter search using WandB sweeps.

Parameters tuned:

Number of epochs (5, 10)

Number of hidden layers (3, 4, 5)

Hidden layer size (32, 64, 128)

Weight decay (0, 0.0005, 0.5)

Learning rate (1e-3, 1e-4)

Optimizer (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam)

Batch size (16, 32, 64)

Weight initialization (Random, Xavier)

Activation functions (Sigmoid, Tanh, ReLU)

Used Bayesian Optimization for an efficient hyperparameter search.

Generated performance plots:

Validation Loss vs. Epochs

Validation Accuracy vs. Epochs

Training Loss vs. Epochs

Training Accuracy vs. Epochs

Results

Best hyperparameter configuration was found using WandB sweeps.

Model achieved high validation accuracy with optimal hyperparameters.

Graphs show trends in loss and accuracy over training epochs.

How to Run

Install dependencies:

pip install numpy keras wandb

Run training script:

python train.py

Run hyperparameter tuning with WandB:

python sweep.py

Conclusion

This assignment successfully implements a feedforward neural network from scratch, with flexible architecture, multiple optimization methods, and efficient hyperparameter tuning using WandB. The model performs well on the Fashion-MNIST dataset and can be further improved with deeper architectures and advanced optimizers.
