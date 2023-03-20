import argparse

def get_train_args():
    parser = argparse.ArgumentParser(description='Training a PyTorch Model')
        
    # add the command-line arguments
    parser.add_argument('data_dir', type=str, help='path to data directory')
    parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121', 'alexnet'],
                        help='model architecture to use (default: vgg16)')
    parser.add_argument("--hidden_layer1_size", type=float, 
                        help="The size of the first hidden layer in the neural network.")
    parser.add_argument("--hidden_layer2_size", type=float, 
                            help="The size of the second hidden layer in the neural network.")
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train the model (default: 15)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--gpu', default=True, type = bool, choices = [True, False], help='Whether to enable GPU usage')
    parser.add_argument('--save_dir', type=str, help='path to save model checkpoints')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for the optimizer')
    
    return parser

