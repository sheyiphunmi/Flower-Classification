import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np
import json

from image_processing import process_image


def load_checkpoint(path, model_class=None, freeze_layers=False):
    
    """
    Loads a PyTorch model from a saved checkpoint.

    Args:
        path (str): The file path of the saved checkpoint.
        model_class (class): The class object of the PyTorch model to load from the checkpoint 
                             (e.g models.vgg16, models.desnenet121, etc). 
                             If None, assume it's the same as the one used to train the checkpoint.
        freeze_layers (bool): If True, freeze all parameters except for the final classifier.

    Returns:
        model: The PyTorch model loaded from the checkpoint.
    """
    try:
        # Load the saved checkpoint from the file at the specified path
        checkpoint = torch.load(path, map_location= 'cpu')
    except Exception as e:
        raise type(e)("Error loading the checkpoint from file: {}".format(path)) from e 
        #using 'from e' to the above helps preserves the original exception traceback and context but also adds a customised message to the exception
        
    if not model_class:
        # If model_class is not provided, assume it's  same as that used to train the checkpoint
        model_class = getattr(models, checkpoint['network'])   # models are from torchvision
    
    try:
        # Create a new instance of the PyTorch model
        model = model_class(pretrained=True)
    except:
        raise ValueError("Invalid model architecture class: {}".format(model_class))
        
    try:
        # Update the state of the new model instance to match the saved state
        model.classifier = checkpoint['classifier']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        model.optimizer = checkpoint['optimizer']
        model.epochs = checkpoint['epochs']
        
        if freeze_layers:
            # Freeze all parameters except for the final classifier
            for param in model.parameters():
                param.requires_grad = False   # Optimizer will not compute gradient and parameter update
            for param in model.classifier.parameters():
                param.requires_grad = True
    except:
        raise ValueError("Error updating model state from checkpoint file: {}".format(path))
        
    # Return the updated model instance
    return model


def predict(image_path, model, top_k, gpu):
    ''' 
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
    - image_path: file path of the image to be predicted
    - model: trained deep learning model
    - top_k: number of top most likely classes to be returned
    - gpu: specifies whether to enable GPU usage (True or Fasle). Default = True
    
    Returns:
    - probs: a list of top probabilities for each class in descending order of magnitude
    - classes: a list of corresponding top classes predicted for the image. Same order as probs
    '''
    
    # Check if CUDA is available and if the user wants to use GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() and gpu == True else "cpu")
    
    # Preprocess the image
    # Move tensor image to device
    img = process_image(image_path)
    img = img.unsqueeze_(0)
    img = img.to(device)
   
    # calculate the class probabilities 
    # Set the model to evaluation mode and turn off gradients
    model.eval()
    with torch.no_grad():
        
        # move the model to the device
        model.to(device)
        
        # Make predictions
        output = model(img)
        probabilities = torch.exp(output)
        top_probabilities, top_classes = probabilities.topk(top_k, dim=1)

    # Convert tensor to numpy arrays
    # The `.cpu()' method is used to move tensors from the GPU to the CPU
    top_probabilities = top_probabilities.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]

    # Mapping indices to class labels
    idx_to_class = {idx: key for key, idx in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in top_classes]
    
    return top_probabilities, top_classes

def read_json_file(filename):
    """
    Reads a JSON file and returns its contents as a Python dictionary.

    Args:
    - filename (str): The name of the JSON file to read.

    Returns:
    - data (dict): The contents of the JSON file as a Python dictionary.
    """
    with open(filename, 'r') as f:
        data = json.load(f)
    return data