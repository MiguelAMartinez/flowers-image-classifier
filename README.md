## Deep Learning with PyTorch - Image Classifier

### Project Overview
In this project I use deep learning with PyTorch to make an image classifier that predicts the top K flower classes and their associated probabilities from a picture.  

### Data
This project uses the [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) from the University of Oxford. It consist of 102 categories of flowers, each containing 40 to 258 images.

### Project Files
The `image_classifier.ipynb` file contains the Jupyter Notebook for the design, training, testing and evaluation of the deep learning model.

The files `model_ic.py`, `utils_ic.py`, `train.py`, `predict.py` convert the model into a command line application. `train.py` trains a new network on a dataset and saves the model as a checkpoint. `predict.py` uses a trained network to predict the class for an input image.

I completed this project as a part of the Udacity Data Scientist Nanodegree program. 
