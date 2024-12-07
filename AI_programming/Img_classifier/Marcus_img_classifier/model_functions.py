import torch
from torchvision import datasets, models, transforms
from torch import nn, optim
from functions import Classifier

def dataset(path):
    '''
    Transforms and creates a datasets
    
    --------------------
    Args:
    -path: The path where the data set is located
    '''
    train_dir = path + '/train'
    valid_dir = path + '/valid'
    test_dir = path + '/test'
    
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])
                                      ])

    valid_transform = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])
                                         ])

    test_transform = transforms.Compose([transforms.Resize(225),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
                                        ])

    # Done: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform = train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_transform )
    test_dataset = datasets.ImageFolder(test_dir, transform = test_transform)

    # Done: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 32)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 32)
    
    return trainloader, testloader, validloader
    
def train(models, learn_rate, epoch, device, trainloader, testloader, print_every=5):
    """
    Train a PyTorch model using a specified learning rate and number of epochs.
    ---------------------------------------------------------------
    
    Args:
    - models: PyTorch model to be trained
    - learn_rate: Learning rate for the optimizer
    - epoch: Number of epochs for training
    - device: Device to use for training (e.g., 'cuda' or 'cpu')
    - trainloader: DataLoader for training dataset
    - testloader: DataLoader for testing dataset
    - print_every: Number of steps to print training progress

    Returns:
    - model: Trained PyTorch model
    """
    torch.cuda.empty_cache()

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    if models.architecture == 'resnet50':
        optimizer = optim.Adam(models.fc.parameters(), lr=learn_rate)
    else:
        optimizer = optim.Adam(models.classifier.parameters(), lr=learn_rate)
        

    steps = 0
    running_loss = 0
    print('\tTraining Commenced')
    for epochs in range(epoch):
        for inputs, labels in trainloader:
            steps += 1

            # Move input and label tensors to the specified device
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            logps = models(inputs)
            loss = criterion(logps, labels)

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()

            # Print training progress
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0

                # Evaluate model on the test set
                models.eval()

                with torch.no_grad():
                    for test_inputs, test_labels in testloader:
                        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
                        logps = models.forward(test_inputs)
                        batch_loss = criterion(logps, test_labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == test_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Print training and testing metrics
                print(f"Epoch {epochs + 1}/{epoch}.. "
                      f"Train loss: {running_loss / print_every:.3f}.. "
                      f"Test loss: {test_loss / len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy / len(testloader):.3f}")

                # Reset running loss and switch back to training mode
                running_loss = 0
                models.train()
    print('\tTraining Concluded')
    return models


def create_model(architecture, hidden_layers, output=102, drop_probability=0.2):
    """
    Create a custom model based on the specified architecture.
    ---------------------------------------------------------------
    
    Args:
    - architecture (str): The architecture of the model ('resnet50', 'vgg13', or 'vgg16').
    - hidden_layers (list): List of sizes for hidden layers in the classifier.
    - output (int): Number of output classes.
    - drop_probability (float): Dropout probability for the classifier.

    Returns:
    - model: Customized PyTorch model.
    """
    drop_prob = drop_probability

    if architecture == 'resnet50':
        resnet_model = models.resnet50(pretrained=True)
        # Save the model name as an attribute of the model, needed while saving the model
        resnet_model.architecture = 'resnet50'

        for param in resnet_model.parameters():
            param.requires_grad = False

        # Get the number of input features for the classifier
        input_size = resnet_model.fc.in_features

        # Create a custom classifier
        classifier = Classifier(input_size, hidden_layers, output, drop_probability=drop_prob)
        resnet_model.fc = classifier
        return resnet_model

    elif architecture == 'vgg13':
        vgg13_model = models.vgg13(pretrained=True)
        # Save the model name as an attribute of the model, needed while saving the model
        vgg13_model.architecture = 'vgg13'

        for param in vgg13_model.parameters():
            param.requires_grad = False

        # Get the number of input features for the classifier
        input_size = vgg13_model.classifier[0].in_features

        # Create a custom classifier
        classifier = Classifier(input_size, hidden_layers, output, drop_probability=drop_prob)
        vgg13_model.classifier = classifier
        return vgg13_model

    elif architecture == 'vgg16':
        vgg16_model = models.vgg16(pretrained=True)
        # Save the model name as an attribute of the model, needed while saving the model
        vgg16_model.architecture = 'vgg16'

        for param in vgg16_model.parameters():
            param.requires_grad = False

        # Get the number of input features for the classifier
        input_size = vgg16_model.classifier[0].in_features

        # Create a custom classifier
        classifier = Classifier(input_size, hidden_layers, output, drop_probability=drop_prob)
        vgg16_model.classifier = classifier
        return vgg16_model

if __name__ == '__main__':
    print('No Errors Encountered')
        
                