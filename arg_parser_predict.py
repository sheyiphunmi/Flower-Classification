import argparse

def get_predict_args():
    parser = argparse.ArgumentParser(description='Predict the possible flower name from an image along with the probability')
    parser.add_argument('image_dir', type=str,help='Path to the image file to be predicted')
    parser.add_argument('checkpoint', type=str,help='Path to the saved checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', default='cat_to_name.json', 
                        help='Path to a JSON file mapping labels to flower names')
    parser.add_argument('--gpu', default=True, type = bool, choices = [True, False], help='Whether to enable GPU usage')
#     parser.add_argument('--arch', type=str, default='vgg16', choices=['vgg16', 'densenet121', 'alexnet'],
#                         help='model architecture to use (default: vgg16)')
    return parser



