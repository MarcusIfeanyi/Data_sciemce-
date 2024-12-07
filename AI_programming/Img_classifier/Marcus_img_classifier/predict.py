from argparse import ArgumentParser, Namespace
import torch
from PIL import Image
from torch import nn
import torch.nn.functional as F
from functions import process_image, Classifier, load_checkpoint, classifier, predict
import json

# Create an argument parser with a description
parser = ArgumentParser(description='Make a Prediction with a saved model')

# Create an argument group for required inputs
group_1 = parser.add_argument_group('input and -checkpoint are required together!')
group_1.add_argument('img_path', type=str, help='Specify path to the image to be predicted')
group_1.add_argument('checkpoint', type=str, help='Specify the path to the model to be used for prediction')

# Add optional arguments
parser.add_argument('--gpu', help='Specify the processing unit for Training', action='store_true')
parser.add_argument('-t', '--top_k', type=int, help='Specify the number of the top most likely classes to be outputed',
                    default=5)
parser.add_argument('-c', '--category_names', type=str,
                    help='Specify the directory to a class-to-index mapping for flower names')

# Parse the command-line arguments
args: Namespace = parser.parse_args()

# Extract values from the arguments
img_path = args.img_path
checkpoint_path = args.checkpoint
gpu_arg = args.gpu
top_k = args.top_k
path_to_json = args.category_names

# Check if GPU is requested and available
if gpu_arg and torch.cuda.is_available():
    print('\tGPU Active...')
    device = 'cuda'
else:
    print('\tCPU Active...')
    device = 'cpu'

# Process the input image
processed_image = process_image(img_path)

# Load the model checkpoint
model = load_checkpoint(checkpoint_path, dev = device)

# Make a prediction
probs, classes = predict(processed_image, model, topk=top_k, dev = device)
print('\tPredicted')
print('Class \tPrediction Percentage\n')

# Display results with or without category names
if path_to_json:
    with open(path_to_json, 'r') as f:
        cat_to_name = json.load(f, strict = False)
    flower_names = [cat_to_name[str(cls)] for cls in classes]
    for flower, prob in zip(flower_names, probs):
        print('{} \t {:0.2f}%'.format(flower, prob * 100))
else:
    for flower, prob in zip(classes, probs):
        print('{} \t {:0.2f}%'.format(flower, prob * 100))
