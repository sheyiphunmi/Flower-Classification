import matplotlib.pyplot as plt

import torch
from torch import nn, optim
import torch.nn.functional as F

import torchvision
from torchvision import datasets, transforms, models

from collections import OrderedDict

import time

from PIL import Image
import numpy as np

import json

from arg_parser_train import get_train_args
import train_helper
from workspace_utils import active_session

with active_session():
    def main():

        
        start_time = time.time()

#         parser = argparse.ArgumentParser(description='Training the Model')

#         # add the command-line arguments
#         parser.add_argument('data_dir', type=str, default = './flowers', help='path to data directory')
#         parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121', 'alexnet'],
#                             help='model architecture to use (default: vgg16)')
#         parser.add_argument("--hidden_layer1_size", type=float, 
#                             help="The size of the first hidden layer in the neural network.")
#         parser.add_argument("--hidden_layer2_size", type=float, 
#                                 help="The size of the second hidden layer in the neural network.")
#         parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model 
#                             (default: 15)')
#         parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
#         parser.add_argument('--enable_gpu', action='store_true', default=True, help='Enable GPU usage')
#         parser.add_argument('--save_dir', type=str, help='path to save model checkpoints')
#         parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the optimizer')

        # parse the command-line arguments
        parser = get_train_args()
        args = parser.parse_args()

        # extract the argument values
        data_dir = args.data_dir
        learning_rate = args.learning_rate
        batch_size = args.batch_size
        arch = args.arch
        save_dir = args.save_dir
        gpu = args.gpu
        hidden_layer1_size = args.hidden_layer1_size
        hidden_layer2_size = args.hidden_layer2_size
        epochs = args.epochs

        #print(args)

        _, _, train_data, testloader, validloader, trainloader = train_helper.load_data(data_dir, batch_size)
        
        print("\nTraining the model\n")
        model, criterion, optimizer, input_features = train_helper.build_model(arch=arch, learning_rate=learning_rate,
                                                                               gpu = gpu,
                                                                               hidden_layer1_size=hidden_layer1_size,
                                                                               hidden_layer2_size=hidden_layer2_size,
                                                                               dropout_probability=0.4
                                                                               )

        train_losses, valid_losses = train_helper.train_model(model, criterion, optimizer, trainloader, 
                                                              validloader, epochs, gpu, validation=True)
#         train_helper.get_hidden_out_features(model)

        train_helper.save_checkpoint(model, arch, input_features, epochs, learning_rate, 
                                     batch_size, optimizer, train_data,save_dir, output_size=102)
    
        # Measure total program runtime by collecting end time
        end_time = time.time()

        # Computes overall runtime in seconds & prints it in hh:mm:ss format
        tot_time = end_time - start_time #calculate difference between end time and start time
        print("\n** Total Elapsed Runtime:",
              str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"
              +str(int((tot_time%3600)%60)) )

    # Call to main function to run the program
    if __name__ == "__main__":
        main()