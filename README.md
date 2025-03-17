# Feedforward Neural Network with Backpropagation from Scratch On Fashion MNIST Dataset

This assignment contains a feedforward neural network from scratch using NumPy. The backpropagation algorithm is implemented without using any automatic differentiation packages. The network is trained on the Fashion-MNIST dataset to classify images into 10 clothing categories. Various optimization algorithms and hyperparameter tuning techniques are also explored.

## Problem Statement
The goal of this assignment is to implement a fully connected neural network and train it using the backpropagation algorithm. The network should be flexible in terms of the number of hidden layers and neurons per layer. The dataset used is Fashion-MNIST, consisting of 60,000 training images and 10,000 test images of size 28x28.

## Code Structure
- `train.py`: Contains the implementation of the feedforward neural network, weight initialization, activation functions, forward propagation, backpropagation, and optimization algorithms.
- `data_processing.py`: Handles data loading, normalization, one-hot encoding, and dataset splitting.
- `optimizers.ipynb`: It has complete answers for asked questions in assignment. 

## Implementation Details
### Data Preprocessing
- Loaded Fashion-MNIST dataset using `keras.datasets.fashion_mnist`.
- Normalized pixel values for better training.
- Converted labels into one-hot encoded format.
- Split dataset into training, validation, and test sets.

### Feedforward Neural Network
- Flexible architecture to allow changes in the number of layers and neurons per layer.
- Supports different activation functions: ReLU, Sigmoid, Tanh for hidden layers, and Softmax for output.
- Supports different weight initialization methods: Random and Xavier.

### Backpropagation
- Implemented in `Back_Propogation()` function.
- Updates weights using gradients computed for each layer.
- Supports multiple loss functions: Cross-Entropy and Mean Squared Error.

### Optimization Algorithms
Each optimization algorithm is implemented as a separate function in `neural_network_fashion_MNIST.py` file:
- **SGD**: `Stochastic_GD()`
- **Momentum-based GD**: `Momentum_GD()`
- **Nesterov accelerated GD**: `Nesterov_GD()`
- **RMSprop**: `RMS_Opt()`
- **Adam**: `Adam_Opt()`
- **Nadam**: `NAdam_Opt()`

### Hyperparameter Tuning using WandB
- Used `wandb.sweep` for hyperparameter optimization.
- Explored different values for epochs, hidden layers, hidden layer sizes, weight decay, learning rate, optimizer, batch size, weight initialization, and activation functions.
- Applied Bayesian Optimization to efficiently find the best hyperparameters.
- Logged and visualized results using WandB.

## Results and Analysis
- Plotted **Validation Loss vs. Epochs**, **Validation Accuracy vs. Epochs**, **Training Loss vs. Epochs**, and **Training Accuracy vs. Epochs**.
- Analyzed different hyperparameter settings to find the best performing configuration.

## How to Run
1. Install required dependencies:
   ```
   pip install numpy tensorflow matplotlib seaborn scikit-learn wandb
   ```
2. Download
   ```
   neural_network_fashion_MNIST.py
   train.py
   ```
   
   files and keep it in same folder
4. Initialize Weights & Biases:
   ```
   wandb login
   ```

5. Train the model: Type one of this below command in command line interface (CLI)
   ```
   python train.py --wandb_entity myname --wandb_project myprojectname
   python train.py
   ```
   you will able to run the code for default parameters.


## Acknowledgments
This project was developed as part of an academic assignment DA6401 for training a neural network from scratch using backpropagation on the Fashion-MNIST dataset.
