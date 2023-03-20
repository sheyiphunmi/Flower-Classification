
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import time

from PIL import Image


def load_data(data_dir, batch_size):

    """Load image datasets and create data loaders.

    Args:
        data_dir (str): Path to the data directory.
        batch_size (int): Batch size for loading the data.

    Returns:
        tuple: A tuple of three image folder and three data loaders for train, test, and validation datasets.

    """
    # Define directories for train, validation, and test datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define transformations for the datasets
    cropped_size = 224
    resized_size = 255
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    rotate = 30

    # Transforms for training dataset:
    # Random scaling, cropping, and flipping, rotation to 30, resized to 224x224 pixels
    train_transforms = transforms.Compose([transforms.RandomRotation(rotate),
                                           transforms.RandomResizedCrop(cropped_size),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])

    # Transforms for test dataset:
    # Resize, center crop to 224x224 pixels, and normalize
    test_transforms = transforms.Compose([transforms.Resize(resized_size),
                                          transforms.CenterCrop(cropped_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])

    # Transforms for validation dataset:
    # Resize, center crop to 224x224 pixels, and normalize
    valid_transforms = transforms.Compose([transforms.Resize(resized_size),
                                           transforms.CenterCrop(cropped_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, stds)])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms) 
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms) 

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle = True)

    # Print the number of images and batches in each dataset
    print(f"Train data: {len(train_data)} images / {len(trainloader)} batches")
    print(f"Test data: {len(test_data)} images / {len(testloader)} batches")
    print(f"Valid data: {len(valid_data)} images / {len(validloader)} batches")

    # Return the data loaders for train, test, and validation datasets
    return test_data, valid_data, train_data, testloader, validloader, trainloader



def build_model(arch, learning_rate, gpu, hidden_layer1_size=None, hidden_layer2_size=None, dropout_probability=0.4):
    """
    Builds a new neural network model with the specified architecture and hyperparameters.

    Args:
        arch (str): the name of the pre-trained model architecture to use (default: 'vgg16')
        hidden_layer1_size (float): the number of neurons in the first hidden layer (default: None)
        hidden_layer2_size (float): the number of neurons in the second hidden layer (default: None)
        dropout_probability (float): the probability of dropout regularization (default: 0.4)
        learning_rate (float): the learning rate for the optimizer (default: learning_rate)
        gpu: whether to use GPU for training (default: True)

    Returns:
        tuple: a tuple containing the model, criterion, optimizer, and input_features
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True  else "cpu")

    # Load pre-trained model
    try:
        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
        elif arch == 'alexnet':
            model = models.alexnet(pretrained=True)
        elif arch == 'densenet121':
            model = models.densenet121(pretrained=True)
        else:
            raise ValueError(f"{arch} is not a valid architecture or not one of the choices. Use either vgg16, alexnet, or densenet121")
    except ValueError as e:
        print(e)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define the number of outputs
    output_size = 102
    
    # Define the number of inputs
    if arch == 'vgg16':
        input_features = model.classifier[0].in_features
    elif arch == 'alexnet':
        input_features = model.classifier[1].in_features
    elif arch == 'densenet121':
        input_features = model.classifier.in_features

    # Define the number of hidden units in the first and second layers
    if hidden_layer1_size is not None and hidden_layer2_size is not None:
        hidden_layer1 = hidden_layer1_size
        hidden_layer2 = hidden_layer2_size
    else:
        if arch == 'vgg16' or arch == 'alexnet':
            hidden_layer1 = 2048
            hidden_layer2 = 512
        elif arch == 'densenet121':
            hidden_layer1 = 560
            hidden_layer2 = 256

    # Define the model architecture
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_features, hidden_layer1)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=dropout_probability)),
        ('fc2', nn.Linear(hidden_layer1, hidden_layer2)),
        ('relu2', nn.ReLU()),
        ('dropout2', nn.Dropout(p=dropout_probability)),
        ('output', nn.Linear(hidden_layer2, output_size)),
        ('log_softmax', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)
    
    return model, criterion, optimizer, input_features




def train_model(model, criterion, optimizer, trainloader, validloader, epochs, gpu, validation=True):


    """
    Trains a PyTorch model and prints out training and validation loss and accuracy.
    
    Args:
    - model (nn.Module): the PyTorch model to train
    - criterion (nn.Module): the loss function to use for training
    - optimizer (torch.optim.Optimizer): the optimizer to use for training
    - trainloader (DataLoader): the DataLoader for the training set
    - validloader (DataLoader): the DataLoader for the validation set
    - gpu (bools): whether to use GPU for training (default: True)
    - arch (str): the name of the architecture to use (default: 'vgg16')
    - epochs (int): the number of epochs to train for (default: 15)
    - validation (bool): whether to perform validation after each epoch (default: True)
    
    Returns:
    - train_losses (list): a list of training losses for each epoch
    - valid_losses (list): a list of validation losses for each epoch
    """
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
    
    train_losses, valid_losses = [], []
    
    for e in range(epochs):
 

        train_loss = 0
        
        # Set model to training mode
        model.train()
        
        for inputs, labels in trainloader:
            # Move input and label tensors to the GPU
            inputs, labels = inputs.to(device), labels.to(device)

            log_pred = model(inputs)
            loss = criterion(log_pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if validation:
            valid_loss = 0
            correct_pred = 0
            
            # Set model to evaluation mode
            model.eval()

            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    log_pred = model(inputs)
                    loss = criterion(log_pred, labels)
                    valid_loss += loss.item()

                    pred = torch.exp(log_pred)
                    top_p, top_class = pred.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    correct_pred += torch.mean(equals.type(torch.FloatTensor)).item()

            # Set model back to training mode
            model.train()

            # Get mean loss to enable comparison between train and test sets
            train_loss = train_loss / len(trainloader)
            valid_loss = valid_loss / len(validloader)

            # At completion of epoch
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)


            # Obtain the accuracy at the completion of the epochs.
            # The fraction of the correct predictions in the test data
            accuracy = correct_pred/len(validloader)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_loss),
                  "Validation Loss: {:.3f}.. ".format(valid_loss),
                  "Validation Accuracy: {:.3f}%".format(accuracy*100))


    return train_losses, valid_losses

def plot_losses(train_losses, valid_losses):
    """
    Plots the training and validation losses over time.

    Args:
    - train_losses (list): a list of training losses for each epoch
    - valid_losses (list): a list of validation losses for each epoch
    """
    plt.plot(train_losses, label='Training loss')
    plt.plot(valid_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


    
def get_hidden_out_features(model):
    
    """
    Returns the number of output features for all hidden layers in the model.

    Args:
    - model(nn.Module): model to get hidden layer output features from

    Returns:
    - list of integers representing the number of output features for all hidden layers in the model
    """
    
#     for layer in model.classifier:
#         if isinstance(layer, nn.Linear):
#             if layer == model.classifier[-2]:
#                 break
#             print(layer.out_features)
    
    return [layer.out_features for layer in model.classifier 
            if isinstance(layer, nn.Linear) and layer != model.classifier[-2]]


def save_checkpoint(model, arch, input_features, epochs, learning_rate, batch_size, optimizer, train_data, save_dir='', output_size=102):
    """
    Saves a checkpoint of the model to a file.

    Args:
    - model (nn.Module): trained model to save
    - arch (str): architecture name of the model
    - input_features (int): number of input features to the model
    - epochs (int): number of epochs trained
    - learning_rate (float): learning rate used for training
    - batch_size (int): batch size used for training
    - optimizer (torch.optim.Optimizer): optimizer used for training
    - train_data (torch.utils.data.Dataset): training dataset used for training the model
    - save_dir (str): directory where to save the checkpoint file (default: current directory)
    - output_size (int): number of output classes (default: 102)
    """

    model.class_to_idx = train_data.class_to_idx

    # Create the checkpoint dictionary
    checkpoint = {
        'network': arch,
        'input_features': input_features,
        'hidden_layers': get_hidden_out_features(model),
        'output': output_size,
        'learning_rate': learning_rate,       
        'batch_size': batch_size,
        'classifier': model.classifier,
        'epochs': epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
#     device = torch.device("cuda" if torch.cuda.is_available() and gpu  else "cpu")

    # If the model is on the GPU, move the checkpoint to the CPU
    #if device.type == 'cuda':
    if next(model.parameters()).is_cuda:
        # Move the model to the CPU
        model.to('cpu')

        # Use a dictionary comprehension to move any tensors in the checkpoint to the CPU
        checkpoint = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in checkpoint.items()}

    # Set the file path for saving the checkpoint
    if save_dir:
        checkpoint_path = f'{save_dir}/checkpoint.pth'
    else:
        checkpoint_path = 'checkpoint.pth'

    # Save the checkpoint to a file
    torch.save(checkpoint, checkpoint_path)
  
    
# def save_checkpoint(model, arch, input_features, epochs, learning_rate, batch_size, optimizer, train_data, save_dir, output_size = 102):
#     model.class_to_idx = train_data.class_to_idx

#     # Create the checkpoint dictionary
#     checkpoint = {'network': arch,
#                 'input_features': input_features,
#                 'hidden_layers': get_hidden_out_features(model),
#                 'output': output_size,
#                 'learning_rate': learning_rate,       
#                 'batch_size': batch_size,
#                 'classifier' : model.classifier,
#                 'epochs': epochs,
#                 'optimizer': optimizer.state_dict(),
#                 'state_dict': model.state_dict(),
#                 'class_to_idx': model.class_to_idx}

#     # If the model is on the GPU, move the checkpoint to the CPU
#     if device.type == 'cuda':
#         # Move the model to the CPU
#         model.to('cpu')

#         # Use a dictionary comprehension to move any tensors in the checkpoint to the CPU
#         checkpoint = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in checkpoint.items()}

#     # Save the checkpoint to a file
#     if args.save_dir:
#         torch.save (checkpoint, args.save_dir + '/checkpoint.pth')
#     else:
#         torch.save (checkpoint, 'checkpoint.pth')


