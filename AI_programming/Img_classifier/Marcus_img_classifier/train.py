import json
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from model_functions import dataset, train, create_model
from functions import save_checkpoint

# Create an argument parser with a description
parser = ArgumentParser(description='Trains a model')

# Add command-line arguments
parser.add_argument('data_dir', metavar='Data directory', type=str, help='Specify training directory')
parser.add_argument('-s', '--save_dir', help='Specify the directory to save the model. The Path and filename', type=str)
parser.add_argument('-lr', '--learning_rate', type=float, help='Specify the model\'s learning rate', default=0.001)
parser.add_argument('--gpu', help='Specify the processing unit for Training', action='store_true')
parser.add_argument('--epochs', type=int, help='Specify the number of training cycles --> Epochs', default=1)
parser.add_argument('--hidden_units', type=int, help='Specify the hidden units', metavar='hidden_layers', nargs='+',
                    default=[500])
parser.add_argument('--arch', type=str, help='Specify the architecture to be used for training', default='resnet50',
                    choices=['resnet50', 'vgg13', 'vgg16'])

# Parse the command-line arguments
args: Namespace = parser.parse_args()

# Extract values from the arguments
data_dir = args.data_dir
save_dir = args.save_dir
gpu_arg = args.gpu
learning_rate = args.learning_rate
epochs = args.epochs
hidden_layer = args.hidden_units
arch = args.arch

# Load dataset
train_loader, test_loader, validloader = dataset(data_dir)

# Check if GPU is requested and available
if gpu_arg and torch.cuda.is_available():
    print('\t\t GPU Active...')
    dev = 'cuda:0'
else:
    print('\t CPU Active...')
    dev = 'cpu'

# Create the model
model = create_model(arch, hidden_layers=hidden_layer)
model.to(dev)

# Train the model
model = train(models=model, epoch=epochs, learn_rate=learning_rate, trainloader=train_loader, testloader=test_loader,
              device=dev)

# Set class-to-index mapping for the model
model.class_to_idx = train_loader.dataset.class_to_idx

# Save the model if a save directory is specified
if save_dir:
    save_checkpoint(models=model, epoch=epochs, path=save_dir)
    print('\tModel saved')
