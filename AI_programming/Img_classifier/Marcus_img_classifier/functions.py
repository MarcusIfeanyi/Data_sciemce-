import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, models, transforms


class Classifier(nn.Module):
    def __init__(self, inputs, hidden_layers, output, drop_probability=0.2):
        '''Creates a feedforward network
        ------------------------------------------------------
        inputs: The input size
        hidden_layers: List containing the hidden layers
        output: The output size of the network
        drop_probability: Dropout probability for the nn.Dropout() (default is 0.2)'''
        super(Classifier, self).__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(inputs, hidden_layers[0])])
        size = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_in, h_out) for h_in, h_out in size])
        self.output = nn.Linear(hidden_layers[-1], output)
        self.dropout = nn.Dropout(p=drop_probability)
        
    def forward(self, z):
        '''Performs a forward pass through the network
        z: Input parameters'''
        for layer in self.hidden_layers:
            z = F.relu(layer(z))
            z = self.dropout(z)
        z = self.output(z)
            
        return F.log_softmax(z, dim=1)

def classifier(architecture, inputs, hidden_layers, output, drop_probability=0.2):
    """
    Create a custom classifier based on the specified architecture.
    
    --------------------------------------------------------------------------

    Args:
    - architecture (str): The architecture of the model ('vgg13', 'vgg16', or 'resnet50').
    - inputs (int): The number of input features.
    - hidden_layers (list): List of sizes for hidden layers in the classifier.
    - output (int): Number of output classes.
    - drop_probability (float): Dropout probability for the classifier.

    Returns:
    - model: Customized PyTorch model.
    """

    if 'vgg13' in architecture:
        vgg13_model = models.vgg13(pretrained=True)
        for param in vgg13_model.parameters():
            param.requires_grad = False
                
        vgg13_model.classifier = Classifier(inputs, hidden_layers, output, drop_probability)
        return vgg13_model
    
    elif 'vgg16' in architecture:
        vgg16_model = models.vgg16(pretrained=True)
        for param in vgg16_model.parameters():
            param.requires_grad = False
                
        vgg16_model.classifier = Classifier(inputs, hidden_layers, output, drop_probability)
        return vgg16_model
    
    elif 'resnet50' in architecture:
        resnet_model = models.resnet50(pretrained=True)
        for param in resnet_model.parameters():
            param.requires_grad = False
        
        resnet_model.fc = Classifier(inputs, hidden_layers, output, drop_probability)
        
        return resnet_model

         
def load_checkpoint(filepath, dev):
    ''' Retrieves the saved model with all its architecture
    ------------------------------------------------
    filepath: path to the saved model
    dev: device to move the model and its components to
    '''
    if torch.cuda.is_available():
        checkpoint = torch.load(filepath, map_location='cuda:0')
    else:
        checkpoint = torch.load(filepath)
    
    # Create the model on the specified device
    model = classifier(checkpoint['arch'],
                       checkpoint['input_size'],
                       checkpoint['hidden_layers'],
                       checkpoint['output_size']).to(dev)
    
    # Move the state dict to the specified device
    model.load_state_dict(checkpoint['state_dict'])
    
    # Move class-to-index mapping to the device
    if 'class_to_idx' in checkpoint:
        model.class_to_idx = {k: torch.tensor(v).to(dev) for k, v in checkpoint['class_to_idx'].items()}
    
    return model

def save_checkpoint(models, epoch, path):
    """
    Save a checkpoint of the model.

    Args:
    - models: The PyTorch model to save.
    - epoch: Number of epochs during training.
    - path (str): Directory path to save the checkpoint.
    - traindataset: The training dataset to get class-to-index mapping.
    """
    # Extract hidden layer sizes and input size based on the model's architecture
    if 'resnet' in models.architecture:
        hidden_layers = [layer.out_features for layer in models.fc.hidden_layers]
        inputs = [layer.in_features for layer in models.fc.hidden_layers if isinstance(layer, torch.nn.Linear)]
        output_size = models.fc.output.out_features
    else:
        hidden_layers = [layer.out_features for layer in models.classifier.hidden_layers]
        inputs = [layer.in_features for layer in models.classifier.hidden_layers if isinstance(layer, torch.nn.Linear)]
        output_size = models.fc.classifier.out_features

    # Create the checkpoint dictionary
    checkpoint = {
        'arch': models.architecture,
        'class_to_idx': models.class_to_idx,
        'state_dict': models.state_dict(),
        'epochs': epoch,
        'hidden_layers': hidden_layers,
        'input_size': inputs[0],
        'output_size': output_size
    }

    # Save the checkpoint to the specified path
    path_to_save = f"{path}/{models.architecture}_model.pth"
    torch.save(checkpoint, path_to_save)
    return 

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a PyTorch tensor with shape [1, 3, 224, 224]
    '''
    # Load the image
    image = Image.open(image_path)

    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Apply the transformations
    img = preprocess(image)
    # Add a batch dimension to the tensor
    img = img.unsqueeze(0)
    return img


def predict(image, model, topk, dev):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Move model and image to the same device
    model.to(dev)
    image = image.to(dev)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
        
        # Convert indices to class labels using the model's class_to_idx mapping
        idx_to_class = {str(idx): str(cls) for idx, cls in model.class_to_idx.items()}
        
        idx_to_class = {str(a): b for a, b in model.class_to_idx.items()}
        
        classes = [idx_to_class[str(idx.item())] for idx in top_class[0]]
        
        return top_p[0].tolist(), classes 

if __name__ == '__main__':
    print('No Errors Encountered')