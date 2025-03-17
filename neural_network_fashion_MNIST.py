
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.datasets import mnist
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


import wandb
#wandb.login()
wandb.login(key = "acdc26d2fc17a56e83ea3ae6c10e496128dee648")

import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Initialize WandB
wandb.init(project="Vinod_FashionMNIST", name="One_Image_Per_Class", reinit=True)

# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Define class labels
fmnist_labels = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Select **one unique image per class**
samples = []
for class_idx in range(10):
    index = np.where(y_train == class_idx)[0][0]  # Get first occurrence of class
    image = wandb.Image(x_train[index], caption=fmnist_labels[class_idx])
    samples.append(image)

# Log **all 10 images in one step**
wandb.log({"Fashion MNIST Classes": samples})

# Finish WandB run
wandb.finish()

# Load and preprocess dataset
def Data_Preprocess(): ##fashon-mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Split into train and validation
    val_size = 5000
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]

    # Normalize dataset
    x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

    # One-hot encoding
    y_train = to_categorical(y_train, 10)
    y_val = to_categorical(y_val, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_val, y_val, x_test, y_test




# Load data
x_train, y_train, x_val, y_val, x_test, y_test = Data_Preprocess()



# Define necessary functions
def Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size):
    weights = {}
    prev_size = input_size
    hidden_size = num_nodes_hidden_layers[0] if isinstance(num_nodes_hidden_layers, list) else num_nodes_hidden_layers

    for i in range(num_hidden_layer):
        if weight == 'random':
            weights[f'W{i+1}'] = np.random.randn(prev_size, hidden_size) * 0.01
        elif weight == 'xavier':
            weights[f'W{i+1}'] = np.random.randn(prev_size, hidden_size) * np.sqrt(1 / prev_size)
        weights[f'b{i+1}'] = np.zeros((1, hidden_size))
        prev_size = hidden_size

    if weight == 'random':
        weights['W_out'] = np.random.randn(prev_size, output_size) * 0.01
    elif weight == 'xavier':
        weights['W_out'] = np.random.randn(prev_size, output_size) * np.sqrt(1 / prev_size)
    weights['b_out'] = np.zeros((1, output_size))

    return weights

def Activation_Function(Z, activation):
    if activation == 'sigmoid':
        return 1 / (1 + np.exp(-Z))
    elif activation == 'tanh':
        return np.tanh(Z)
    elif activation == 'relu':
        return np.maximum(0, Z)
    elif activation == 'softmax':
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

def Derrivative_Activation(Z, activation):
    if activation == 'sigmoid':
        sig = Activation_Function(Z, 'sigmoid')
        return sig * (1 - sig)
    elif activation == 'tanh':
        return 1 - np.tanh(Z)**2
    elif activation == 'relu':
        return (Z > 0).astype(float)

def Cross_Entropy_Loss(y_actual, y_pred):
    return -np.mean(y_actual * np.log(y_pred + 1e-9))


def MSE_Loss(y_actual, y_pred):
    return np.mean((y_actual - y_pred) ** 2)


def Forward_Propogation(X, weights, num_hidden_layer, activation):
    A = X.reshape(X.shape[0], -1)
    cache = {'A0': A}

    for i in range(num_hidden_layer):
        Z = np.dot(A, weights[f'W{i+1}']) + weights[f'b{i+1}']
        A = Activation_Function(Z, activation)
        cache[f'Z{i+1}'] = Z
        cache[f'A{i+1}'] = A

    Z_out = np.dot(A, weights['W_out']) + weights['b_out']
    A_out = Activation_Function(Z_out, 'softmax')

    cache['Z_out'] = Z_out
    cache['A_out'] = A_out

    return A_out, cache

def Back_Propogation(X, y_actual, weights, cache, num_hidden_layer, activation):
    gradients = {}
    m = X.shape[0]

    dZ_out = cache['A_out'] - y_actual
    gradients['dW_out'] = np.dot(cache[f'A{num_hidden_layer}'].T, dZ_out) / m
    gradients['db_out'] = np.sum(dZ_out, axis=0, keepdims=True) / m

    dA = np.dot(dZ_out, weights['W_out'].T)

    for i in range(num_hidden_layer, 0, -1):
        dZ = dA * Derrivative_Activation(cache[f'Z{i}'], activation)
        gradients[f'dW{i}'] = np.dot(cache[f'A{i-1}'].T, dZ) / m
        gradients[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True) / m
        dA = np.dot(dZ, weights[f'W{i}'].T)

    return gradients

def Calculate_Accuracy(X, y_actual, weights, num_hidden_layer, activation):
    y_pred, _ = Forward_Propogation(X, weights, num_hidden_layer, activation)
    return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_actual, axis=1))

def Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size,loss_function, input_size, output_size):
    wandb.init(project="stochastic")
    
    print("i am here")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    print("starting epochs")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            y_pred, cache = Forward_Propogation(X_batch, weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, weights, cache, num_hidden_layer, activation)

            for key in weights:
                weights[key] -= lr * gradients[f'd{key}']

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})
    print("ended epochs")
    return weights




# # Set parameters and train
# num_hidden_layer = 4
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 5
# activation = 'sigmoid'

# trained_weights = Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, loss_function='cross_entropy')
# wandb.finish()

# Modify Momentum_GD to use either cross-entropy or MSE loss

def Momentum_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size,momentum, loss_function, input_size, output_size):
    wandb.init(project="momentum")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    velocity = {key: np.zeros_like(value) for key, value in weights.items()}
    print("Its for Momentum")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred, cache = Forward_Propogation(X_batch, weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, weights, cache, num_hidden_layer, activation)

            for key in weights:
                velocity[key] = momentum * velocity[key] - lr * gradients[f'd{key}']
                weights[key] += velocity[key]

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})

    return weights

# Example usage


# Set parameters and train
# num_hidden_layer = 3
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 1
# activation = 'sigmoid'
# trained_weights1 = Momentum_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, momentum=0.9, loss_function='cross_entropy')
# wandb.finish()

def Nesterov_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, momentum, loss_function, input_size, output_size):
    wandb.init(project="nesterov")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    velocity = {key: np.zeros_like(value) for key, value in weights.items()}
    print("Its for Nestrov")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            temp_weights = {key: weights[key] + momentum * velocity[key] for key in weights}
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred, cache = Forward_Propogation(X_batch, temp_weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, temp_weights, cache, num_hidden_layer, activation)

            for key in weights:
                velocity[key] = momentum * velocity[key] - lr * gradients[f'd{key}']
                weights[key] += velocity[key]

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})

    return weights

# Example usage


# # Set parameters and train
# num_hidden_layer = 5
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 1
# activation = 'sigmoid'

# trained_weights2 = Nesterov_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, momentum=0.9, loss_function='cross_entropy')
# wandb.finish()

def RMS_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, beta, epsilon, loss_function, input_size, output_size):
    wandb.init(project="rmsprop")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    cache = {key: np.zeros_like(value) for key, value in weights.items()}
    print("Its for RMS")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred, cache_forward = Forward_Propogation(X_batch, weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, weights, cache_forward, num_hidden_layer, activation)

            for key in weights:
                cache[key] = beta * cache[key] + (1 - beta) * gradients[f'd{key}']**2
                weights[key] -= lr * gradients[f'd{key}'] / (np.sqrt(cache[key]) + epsilon)

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})

    return weights

# Example usage

# # Set parameters and train
# num_hidden_layer = 4
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 1
# activation = 'sigmoid'


# trained_weights3 = RMS_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta=0.9, epsilon=1e-8, loss_function='cross_entropy')
# wandb.finish()

def Adam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, beta1, beta2, epsilon, loss_function, input_size, output_size):
    wandb.init(project="adam")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    m = {key: np.zeros_like(value) for key, value in weights.items()}
    v = {key: np.zeros_like(value) for key, value in weights.items()}
    print("Its for Adam")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred, cache_forward = Forward_Propogation(X_batch, weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, weights, cache_forward, num_hidden_layer, activation)

            for key in weights:
                m[key] = beta1 * m[key] + (1 - beta1) * gradients[f'd{key}']
                v[key] = beta2 * v[key] + (1 - beta2) * (gradients[f'd{key}'] ** 2)
                m_hat = m[key] / (1 - beta1 ** (epoch + 1))
                v_hat = v[key] / (1 - beta2 ** (epoch + 1))
                weights[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})

    return weights

# Example usage

# Set parameters and train
# num_hidden_layer = 4
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 1
# activation = 'sigmoid'


# trained_weights4 = Adam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')
# wandb.finish()

def NAdam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, beta1, beta2, epsilon, loss_function, input_size, output_size):
    wandb.init(project="nadam")
    weights = Weights_Initialization(num_hidden_layer, num_nodes_hidden_layers, weight, input_size, output_size)
    m = {key: np.zeros_like(value) for key, value in weights.items()}
    v = {key: np.zeros_like(value) for key, value in weights.items()}
    print("Its for Nadam")
    for epoch in range(epochs):
        for i in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]
            y_pred, cache_forward = Forward_Propogation(X_batch, weights, num_hidden_layer, activation)
            gradients = Back_Propogation(X_batch, y_batch, weights, cache_forward, num_hidden_layer, activation)

            for key in weights:
                m[key] = beta1 * m[key] + (1 - beta1) * gradients[f'd{key}']
                v[key] = beta2 * v[key] + (1 - beta2) * (gradients[f'd{key}'] ** 2)
                m_hat = m[key] / (1 - beta1 ** (epoch + 1))
                v_hat = v[key] / (1 - beta2 ** (epoch + 1))
                nadam_update = beta1 * m_hat + (1 - beta1) * gradients[f'd{key}']
                weights[key] -= lr * nadam_update / (np.sqrt(v_hat) + epsilon)

        train_acc = Calculate_Accuracy(x_train, y_train, weights, num_hidden_layer, activation)
        val_acc = Calculate_Accuracy(x_val, y_val, weights, num_hidden_layer, activation)

        # Select loss function dynamically
        if loss_function == 'cross_entropy':
            train_loss = Cross_Entropy_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = Cross_Entropy_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])
        elif loss_function == 'mse':
            train_loss = MSE_Loss(y_train, Forward_Propogation(x_train, weights, num_hidden_layer, activation)[0])
            val_loss = MSE_Loss(y_val, Forward_Propogation(x_val, weights, num_hidden_layer, activation)[0])

        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Val Loss={val_loss:.4f}")
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss, "epoch": epoch + 1})

    return weights

# Example usage

# Set parameters and train
# num_hidden_layer = 4
# num_nodes_hidden_layers = [128]
# weight = 'random'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.01
# batch_size = 64
# epochs = 4
# activation = 'sigmoid'

# trained_weights5 = NAdam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')
# wandb.finish()

"""#Question - 4 , 5 , 6"""

# def main():
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'accuracy', 'goal': 'maximize'},
#         'parameters': {
#             'epochs': {'values': [5, 10]},
#             'num_layers': {'values': [3, 4, 5]},
#             'hidden_size': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 0.0005, 0.5]},
#             'learning_rate': {'values': [1e-3, 1e-4]},
#             'optimizer': {'values': ['stochastic', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
#             'batch_size': {'values': [16, 32, 64]},
#             'weight_init': {'values': ['random', 'xavier']},
#             'activation': {'values': ['sigmoid', 'tanh', 'relu']},
#         }
#     }
#     sweep_id = wandb.sweep(sweep_config, project="Vinod_Assgnment1_Question456")
#     wandb.agent(sweep_id, function=train, count=350)

def train():
    wandb.init(project="Vinod_Assignment1_Question456")
    # wandb.init()
    config = wandb.config
    run_name = f"Opt-{config.optimizer}_Layers-{config.num_layers}_HS-{config.hidden_size}_LR-{config.learning_rate}_Batch-{config.batch_size}_Act-{config.activation}"
    wandb.run.name = run_name

    # x_train, y_train, x_val, y_val, _, _ = Data_Preprocess()

    optimizer = config.optimizer

    if optimizer == 'stochastic':
        trained_weights = Stochastic_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
        # trained_weights = Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size)
    elif optimizer == 'momentum':
        trained_weights = Momentum_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
    elif optimizer == 'nag':
        trained_weights = Nesterov_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
    elif optimizer == 'rmsprop':
        trained_weights = RMS_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
    elif optimizer == 'adam':
        trained_weights = Adam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
    elif optimizer == 'nadam':
        trained_weights = NAdam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')

    #wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

    wandb.finish()

if __name__ == "__main__":
    main()

"""#Question - 7"""

def plot_confusion_matrix(y_true, y_pred, config_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {config_name}')

    wandb.log({f"Confusion Matrix - {config_name}": wandb.Image(plt)})
    plt.close()



# def evaluate_best_configs(best_configs):
#     x_train, y_train, x_val, y_val, x_test, y_test = Data_Preprocess()
#     y_test_labels = np.argmax(y_test, axis=1)

#     for config in best_configs:
#         wandb.init(project="Vinod_Assignment1_Question7", name=f"Confusion_Matrix_{config['name']}", reinit=True)

#         if config['optimizer'] == 'rmsprop':
#             trained_weights = RMS_Opt(config['learning_rate'], x_train, y_train, x_val, y_val, config['epochs'], config['activation'], config['num_layers'], config['hidden_size'], config['weight_init'], config['batch_size'], 28*28, 10)
#         elif config['optimizer'] == 'adam':
#             trained_weights = Adam_Opt(config['learning_rate'], x_train, y_train, x_val, y_val, config['epochs'], config['activation'], config['num_layers'], config['hidden_size'], config['weight_init'], config['batch_size'], 28*28, 10)
#         else:
#             raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

#         y_pred_probs, _ = Forward_Propogation(x_test, trained_weights, config['num_layers'], config['activation'])
#         y_pred_labels = np.argmax(y_pred_probs, axis=1)
#         plot_confusion_matrix(y_test_labels, y_pred_labels, config['name'])
#         wandb.finish()


# if __name__ == "__main__":
#     best_configs = [
#     {
#         'name': 'Best_Config_1',
#         'epochs': 10,
#         'num_layers': 5,
#         'hidden_size': 128,
#         'learning_rate': 0.001,
#         'batch_size': 64,
#         'optimizer': 'adam',
#         'weight_decay': 0.5,
#         'weight_init': 'xavier',
#         'activation': 'tanh'
#     },
#     {
#         'name': 'Best_Config_2',
#         'epochs': 10,
#         'num_layers': 4,
#         'hidden_size': 64,
#         'learning_rate': 0.001,
#         'batch_size': 16,
#         'optimizer': 'adam',
#         'weight_decay': 0.0005,
#         'weight_init': 'xavier',
#         'activation': 'relu'
#     }
# ]
# evaluate_best_configs(best_configs)

# import wandb
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
# import pandas as pd

# def plot_confusion_matrix(y_true, y_pred, config_name, class_names):
#     """Logs an interactive confusion matrix to WandB."""
#     cm = confusion_matrix(y_true, y_pred)

#     # Log interactive Confusion Matrix
#     wandb.log({
#         f"Confusion Matrix - {config_name}": wandb.plot.confusion_matrix(
#             probs=None,
#             y_true=y_true,
#             preds=y_pred,
#             class_names=class_names
#         )
#     })


# def evaluate_best_configs(best_configs):
#     x_train, y_train, x_val, y_val, x_test, y_test = Data_Preprocess()
#     y_test_labels = np.argmax(y_test, axis=1)

#     class_names = ['Ankle boot', 'Bag', 'Coat', 'Dress', 'Pullover', 'Sandal', 'Shirt', 'Sneaker', 'T-shirt/top', 'Trouser']

#     for config in best_configs:
#         wandb.init(project="Vinod_Assignment1_Question7_A", name=f"Confusion_Matrix_{config['name']}", reinit=True)

#         if config['optimizer'] == 'rmsprop':
#             trained_weights = RMS_Opt(config['learning_rate'], x_train, y_train, x_val, y_val, config['epochs'], config['activation'], config['num_layers'], config['hidden_size'], config['weight_init'], config['batch_size'], 28*28, 10)
#         elif config['optimizer'] == 'adam':
#             trained_weights = Adam_Opt(config['learning_rate'], x_train, y_train, x_val, y_val, config['epochs'], config['activation'], config['num_layers'], config['hidden_size'], config['weight_init'], config['batch_size'], 28*28, 10)
#         else:
#             raise ValueError(f"Unsupported optimizer: {config['optimizer']}")

#         y_pred_probs, _ = Forward_Propogation(x_test, trained_weights, config['num_layers'], config['activation'])
#         y_pred_labels = np.argmax(y_pred_probs, axis=1)

#         # Log interactive confusion matrix
#         plot_confusion_matrix(y_test_labels, y_pred_labels, config['name'], class_names)

#         wandb.finish()


# if __name__ == "__main__":
#     best_configs = [
#     {
#         'name': 'Best_Config_1',
#         'epochs': 10,
#         'num_layers': 5,
#         'hidden_size': 128,
#         'learning_rate': 0.001,
#         'batch_size': 64,
#         'optimizer': 'adam',
#         'weight_decay': 0.5,
#         'weight_init': 'xavier',
#         'activation': 'tanh'
#     },
#     {
#         'name': 'Best_Config_2',
#         'epochs': 10,
#         'num_layers': 4,
#         'hidden_size': 64,
#         'learning_rate': 0.001,
#         'batch_size': 16,
#         'optimizer': 'adam',
#         'weight_decay': 0.0005,
#         'weight_init': 'xavier',
#         'activation': 'relu'
#     }
# ]
# evaluate_best_configs(best_configs)

# """#Question - 8"""

# def main():
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'accuracy', 'goal': 'maximize'},
#         'parameters': {
#             'epochs': {'values': [5, 10]},
#             'num_layers': {'values': [3, 4, 5]},
#             'hidden_size': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 0.0005, 0.5]},
#             'learning_rate': {'values': [1e-3, 1e-4]},
#             'optimizer': {'values': ['stochastic', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
#             'batch_size': {'values': [16, 32, 64]},
#             'weight_init': {'values': ['random', 'xavier']},
#             'activation': {'values': ['sigmoid', 'tanh', 'relu']},
#         }
#     }
#     sweep_id = wandb.sweep(sweep_config, project="Vinod_Assignment1_Question8_mse")
#     wandb.agent(sweep_id, function=train, count=50)

# def train():
#     wandb.init(project="Vinod_Assignment1_Question8_mse")
#     # wandb.init()
#     config = wandb.config
#     run_name = f"Opt-{config.optimizer}_Layers-{config.num_layers}_HS-{config.hidden_size}_LR-{config.learning_rate}_Batch-{config.batch_size}_Act-{config.activation}"
#     wandb.run.name = run_name

#     # x_train, y_train, x_val, y_val, _, _ = Data_Preprocess()

#     optimizer = config.optimizer

#     if optimizer == 'stochastic':
#         trained_weights = Stochastic_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')
#         # trained_weights = Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size)
#     elif optimizer == 'momentum':
#         trained_weights = Momentum_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')
#     elif optimizer == 'nag':
#         trained_weights = Nesterov_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')
#     elif optimizer == 'rmsprop':
#         trained_weights = RMS_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')
#     elif optimizer == 'adam':
#         trained_weights = Adam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')
#     elif optimizer == 'nadam':
#         trained_weights = NAdam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mse')

#     #wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

#     wandb.finish()

# if __name__ == "__main__":
#     main()

# def main():
#     sweep_config = {
#         'method': 'bayes',
#         'metric': {'name': 'accuracy', 'goal': 'maximize'},
#         'parameters': {
#             'epochs': {'values': [5, 10]},
#             'num_layers': {'values': [3, 4, 5]},
#             'hidden_size': {'values': [32, 64, 128]},
#             'weight_decay': {'values': [0, 0.0005, 0.5]},
#             'learning_rate': {'values': [1e-3, 1e-4]},
#             'optimizer': {'values': ['stochastic', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam']},
#             'batch_size': {'values': [16, 32, 64]},
#             'weight_init': {'values': ['random', 'xavier']},
#             'activation': {'values': ['sigmoid', 'tanh', 'relu']},
#         }
#     }
#     sweep_id = wandb.sweep(sweep_config, project="Vinod_Assignment1_Question8_CrossEntropy")
#     wandb.agent(sweep_id, function=train, count=50)

# def train():
#     wandb.init(project="Vinod_Assignment1_Question8_CrossEntropy")
#     # wandb.init()
#     config = wandb.config
#     run_name = f"Opt-{config.optimizer}_Layers-{config.num_layers}_HS-{config.hidden_size}_LR-{config.learning_rate}_Batch-{config.batch_size}_Act-{config.activation}"
#     wandb.run.name = run_name

#     # x_train, y_train, x_val, y_val, _, _ = Data_Preprocess()

#     optimizer = config.optimizer

#     if optimizer == 'stochastic':
#         trained_weights = Stochastic_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='mscross_entropy')
#         # trained_weights = Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size)
#     elif optimizer == 'momentum':
#         trained_weights = Momentum_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
#     elif optimizer == 'nag':
#         trained_weights = Nesterov_GD(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
#     elif optimizer == 'rmsprop':
#         trained_weights = RMS_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
#     elif optimizer == 'adam':
#         trained_weights = Adam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')
#     elif optimizer == 'nadam':
#         trained_weights = NAdam_Opt(config.learning_rate, x_train, y_train, x_val, y_val, config.epochs, config.activation, config.num_layers, config.hidden_size, config.weight_init, config.batch_size, 28*28, 10, loss_function='cross_entropy')

#     #wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss": val_loss})

#     wandb.finish()

# if __name__ == "__main__":
#     main()

# """#Question- 10"""

# def Data_Preprocess_mnist():
#     (x_train, y_train), (x_test, y_test) = mnist.load_data()

#     # Split into train and validation
#     val_size = 5000
#     x_val, y_val = x_train[:val_size], y_train[:val_size]
#     x_train, y_train = x_train[val_size:], y_train[val_size:]

#     # Normalize dataset
#     x_train, x_val, x_test = x_train / 255.0, x_val / 255.0, x_test / 255.0

#     # One-hot encoding
#     y_train = to_categorical(y_train, 10)
#     y_val = to_categorical(y_val, 10)
#     y_test = to_categorical(y_test, 10)

#     return x_train, y_train, x_val, y_val, x_test, y_test

# x_train_mnist, y_train_mnist, x_val_mnist, y_val_mnist, x_test_mnist, y_test_mnist = Data_Preprocess_mnist()

# # Set parameters and train
# num_hidden_layer = 5
# num_nodes_hidden_layers = [128]
# weight = 'xavier'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.001
# batch_size = 64
# epochs = 10
# activation = 'tanh'

# trained_weights_mnist1 = Adam_Opt(lr, x_train_mnist, y_train_mnist, x_val_mnist, y_val_mnist, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')

# # Set parameters and train
# num_hidden_layer = 4
# num_nodes_hidden_layers = [64]
# weight = 'xavier'
# input_size = 28 * 28  # Flattened image size
# output_size = 10  # Number of classes
# lr = 0.001
# batch_size = 16
# epochs = 10
# activation = 'relu'

# trained_weights_mnist2 = Adam_Opt(lr, x_train_mnist, y_train_mnist, x_val_mnist, y_val_mnist, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')

# # Set parameters and train
# num_hidden_layer = 3
# num_nodes_hidden_layers = [64]
# weight = 'xavier'
# input_size = 28 * 28
# output_size = 10
# lr = 0.001
# batch_size = 32
# epochs = 10
# activation = 'tanh'

# trained_weights_mnist3 = Adam_Opt(lr, x_train_mnist, y_train_mnist, x_val_mnist, y_val_mnist, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')