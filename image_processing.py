import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

def process_image(image_path):
    
    """Function to Scale, crop, and normalize a PIL image for a PyTorch model.

    Args:
        image_path (PIL.Image): The image to be processed.

    Returns:
        torch.Tensor: The processed image as a tensor array.
    """
    # Open the image file as PIL Image
    with Image.open(image_path) as image:
        
        # Define transformations to be applied to the image
        cropped_size = 224
        resized_size = 255
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        image_transforms = transforms.Compose([
            transforms.Resize(resized_size),
            transforms.CenterCrop(cropped_size),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)
        ])

        # Apply the transformation pipeline
        tensor = image_transforms(image)

#         # Convert PyTorch tensor to NumPy array
#         numpy_array = tensor.numpy()

        return tensor

def imshow(image, ax=None, title=None):
    """
    Display image tensor with pyplot.
    
    Args:
    - image (torch.Tensor): Image tensor to display.
    - ax (matplotlib.axes.Axes, optional): Axes object to display the image on.
        If not provided, python creates a new figure and axes.
    - title (str, optional): Title for the displayed image.
    
    Returns:
    - ax (matplotlib.axes.Axes): Axes object displaying the image.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension. Hence, a need to transform the dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1, else, it looks noisy when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax



