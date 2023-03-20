import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

from PIL import Image
import numpy as np
import json

from arg_parser_predict import get_predict_args
import predict_helper 

def main():
    
    parser = get_predict_args()
    args = parser.parse_args()
    
    # extract the argument values
    image_dir = args.image_dir
    checkpoint_path = args.checkpoint
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu
    
    # Load model from checkpoint
    model = predict_helper.load_checkpoint(checkpoint_path, model_class=None, freeze_layers=False)
    
    # Make predictions about the class of an image
    # returns the classes and probabilities in descending order of probabilities
    top_probs, top_classes = predict_helper.predict(image_dir, model, top_k, gpu)
    
    cat_to_name = predict_helper.read_json_file(category_names)
    
#     if category_names:
#         cat_to_name = read_json_file(category_names)
#     else:
#         cat_to_name = read_json_file('cat_to_name.json')
    
    class_names = [cat_to_name[item] for item in top_classes]
    
    # prints the top_k predicted classes of the flowers and their corresponding probability
    for idx in range(top_k):
        print(f"{idx+1}. Class: {class_names[idx]}  Probability: {top_probs[idx]:.3f}")
    
    
# A call to main function to run the program
if __name__ == "__main__":
    main()    

print("\n Prediction Done!")