import argparse
import numpy as np
import wandb
from neural_network_fashion_MNIST import Data_Preprocess, Momentum_GD, Nesterov_GD

parser = argparse.ArgumentParser(description='Neural Network Training Configuration')


parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname', help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we', '--wandb_entity', type=str, default='myname', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
parser.add_argument('-sid', '--wandb_sweepid', type=str, default=None, help='Wandb Sweep Id to log in sweep runs the Weights & Biases dashboard.')
parser.add_argument('-d', '--dataset', type=str, default='fashion_mnist', choices=["mnist", "fashion_mnist"], help='Dataset choices: ["mnist", "fashion_mnist"]')
parser.add_argument('-e', '--epochs', type=int, default=5, help='Number of epochs to train neural network.')
parser.add_argument('-b', '--batch_size', type=int, default=64, help='Batch size used to train neural network.')
parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=["mean_squared_error", "cross_entropy"], help='Loss function choices: ["mean_squared_error", "cross_entropy"]')
parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=["stochastic", "momentum", "nag", "rmsprop", "adam", "nadam"], help='Optimizer choices: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Learning rate used to optimize model parameters')
parser.add_argument('-m', '--momentum', type=float, default=0.9, help='Momentum used by momentum and nag optimizers.')
parser.add_argument('-beta', '--beta', type=float, default=0.99, help='Beta used by rmsprop optimizer')
parser.add_argument('-beta1', '--beta1', type=float, default=0.9, help='Beta1 used by adam and nadam optimizers.')
parser.add_argument('-beta2', '--beta2', type=float, default=0.999, help='Beta2 used by adam and nadam optimizers.')
parser.add_argument('-eps', '--epsilon', type=float, default=1e-08, help='Epsilon used by optimizers.')
parser.add_argument('-w_d', '--weight_decay', type=float, default=0.0005, help='Weight decay used by optimizers.')
parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=["random", "xavier"], help='Weight initialization choices: ["random", "xavier"]')
parser.add_argument('-nhl', '--num_layers', type=int, default=5, help='Number of hidden layers used in feedforward neural network.')
parser.add_argument('-sz', '--hidden_size', type=int, default=128, help='Number of hidden neurons in a feedforward layer.')
parser.add_argument('-a', '--activation', type=str, default='tanh', choices=["identity", "sigmoid", "tanh", "ReLU"], help='Activation function choices: ["identity", "sigmoid", "tanh", "ReLU"]')

args = parser.parse_args()


# train_data, test_data = None, None
# if args.dataset == 'mnist':
#     from keras.datasets import mnist
#     train_data, test_data = mnist.load_data()
# elif args.dataset == 'fashion_mnist':
#     from keras.datasets import fashion_mnist
#     train_data, test_data = fashion_mnist.load_data()



# X_train, y_train = train_data

loss = args.loss
batch_size = args.batch_size
optimizer = args.optimizer
lr = args.learning_rate
momentum = args.momentum
beta = args.beta
epochs = args.epochs
wandb_project = args.wandb_project
wandb_entity = args.wandb_entity
activation = args.activation
num_hidden_layer = args.num_layers
num_nodes_hidden_layers = args.hidden_size
weight = args.weight_init





print(loss,batch_size,optimizer,lr,beta,epochs,wandb_entity,wandb_project)
if args.dataset == 'fashion_mnist':
    from keras.datasets import fashion_mnist

    x_train, y_train, x_val, y_val, x_test, y_test  =Data_Preprocess()

    #weights1 = Stochastic_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size,loss, input_size=28*28, output_size=10);
    #weights2 = Momentum_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, momentum,loss, input_size= 28*28, output_size=10)
    weights3 = Nesterov_GD(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, momentum, loss, input_size = 28*28, output_size=10)
    #weights4 = RMS_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size = 28*28, output_size=10, beta=0.9, epsilon=1e-8, loss_function='cross_entropy')
    #weights5 = Adam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')
    #weights6 = NAdam_Opt(lr, x_train, y_train, x_val, y_val, epochs, activation, num_hidden_layer, num_nodes_hidden_layers, weight, batch_size, input_size, output_size, beta1=0.9, beta2=0.999, epsilon=1e-8, loss_function='cross_entropy')



# else:
#     train_data, test_data = mnist.load_data()


#     x_train, y_train, x_val, y_val, x_test, y_test  = Data_Preprocess_mnist()
    

