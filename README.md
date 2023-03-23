# Flower-Classification

This project is an image classifier implemented in PyTorch that utilizes three different architectures: VGG16, AlexNet, and DenseNet121. It has been trained on a dataset of flower images with 102 different flower categories, enabling it to predict the class of a given flower image. A practical application of this image classifier is the ability to use the trained model for real-time flower identification through a smartphone camera. In addition, the project serves as the final project for [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089), which demosntrates my proficiency in applying AI techniques.


# Installation

1. Clone the repository:

```
git clone https://github.com/sheyiphunmi/Flower-Classification.git
```

2. Install the required packages
 
```
pip install -r requirements.txt
```

# Usage

There are several scripts in this project:

* `arg_parse_train.py`: Defines the command line arguments for training the image classifier.
* `arg_parse_predict.py`: Defines the command line arguments for making predictions with the image classifier.
* `image_processing.py`: Contains functions for preprocessing images before feeding them into the `predict.py`.
* `predict.py`: Loads a saved image classifier model and makes predictions on new images.
* `predict_helper.py`: Contains functions used by `predict.py`.
* `train.py`: Trains an image classifier on the flower dataset.
* `train_helper.py`: Contains functions used by `train.py`

To train the image classifier, run `train.py` with the following command:

```
python train.py flowers --arch <model_architecture> --learning_rate <learning_rate> --hidden_layer1_size <hidden_layer1_size> --hidden_layer2_size <hidden_layer2_size> --epochs <number_of_epochs> --save_directory <path_to_save_checkpoint> --gpu <use_gpu> 
```
* `flowers`: The name of the data directory containing the flower images.
* `data_directory`: The path to the directory containing the flower images.
* `--arch`: PyTorch architectur to use. Choices are VGG16, Alexnet and Densenet121 (Default: `VGG16`).
* `--gpu`: Use GPU for training (default = True).
* `--save_dir`: The directory to save the trained model checkpoint.
* `--learning_rate`: The learning rate for the optimizer (Default: 0.001).
* `--hidden_layer1_size`: The size of the first hidden layer in the neural network (default depends on the `--arch` used and could be found in `train_helper.py)`.
* `--hidden_layer2_size`: The size of the second hidden layer in the neural network (default depends on the `--arch` used and could be found in `train_helper.py)`.
* `--epochs`: The number of epochs to train the model (Default: 15).
* `--batch_size`: The batch size for training (Default: 64)


To make predictions new images of flowers using the trained model, run the `predict.py` using the following command:

```
python predict.py /path/to/image /path/to/checkpoint --top_k <number_of_top_predictions> --category_names <path_to_category_names> --gpu <use_gpu>
```
* `--top_k`: The number of top predictions to display (Default: 5; optional).
* `--category_names`: The path to the JSON file mapping the flower category names to the class indices (optional).
* `--gpu`: Use GPU for inference (optional).

# Dataset

The [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) used to train this image classifier is the [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the Visual Geometry Group at Oxford. Each category consists of between 40 and 258 images.

# Acknowledgement

* This project is my final project for [AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089).
* The flower dataset used in this project was provided by Udacity
* The [102 Category Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the Visual Geometry Group at Oxford.
* The PyTorch documentation and tutorials.

# License

This project is licensed under the [MIT License](https://opensource.org/license/mit/)


