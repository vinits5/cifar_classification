# Cifar Classification

This repository contains CNN for classification of CIFAR-10 dataset.

## Dataset:
Download the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) with python version of size 163 MB by Torronto University.\
This dataset has 10 categories as follows:\
*airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck*\
Copy the cifar-10-batches-py.zip file in this repository and extract here.

## Code

**[train.py](https://github.com/vinits5/cifar_classification/blob/master/train.py)** is main code to train the network.\
**[helper.py](https://github.com/vinits5/cifar_classification/blob/master/helper.py)** has functions to deal with data.\
**[models](https://github.com/vinits5/cifar_classification/blob/master/models/)** contain the network architecture.\
**[read_data.py](https://github.com/vinits5/cifar_classification/blob/master/read_data.py)** is just to show you how to read data and visualize images.

Start the training: *python train.py --mode train*\
Test the network: *python train.py --mode test*

Start training with additional parameters: *python train.py --mode train --args value*

### Parameters:
**Args:				Description**\
**mode:**               train or test\
**model:** classifier or classifier_VGG\
**log_dir:**			store all log data of training.\
**img_size:**			Size of image in CIFAR-10 (32x32x3)\
**channels:**			Channels in image (3 for RGB and 1 for GRAY scale)\
**learning_rate:**		Initial Learning Rate for training\
**decay_rate:**			Decay rate for learning rate\
**decay_step:**			Steps after which learning rate will drop.\
**max_epoch:**			Maximum number of episodes for training.\
**batch_size:**			Size of images used once to train the network.\
**model_path:**			Path of trained log/weights to test the network.\

