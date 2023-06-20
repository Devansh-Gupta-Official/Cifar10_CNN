import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Label	Description
# 0	airplane
# 1	automobile
# 2	bird
# 3	cat
# 4	deer
# 5	dog
# 6	frog
# 7	horse
# 8	ship
# 9	truck

from keras.datasets import cifar10

(X_train,y_train), (X_test,y_test) = cifar10.load_data()
print(X_train.shape)   #50000 samples and each image has 32x32 dimensions and rgb or 3 channels
print(y_train.shape)   #50000 samples and each sample has 1 corresponding claasification like horse, cat, etc...

# print(plt.imshow(X_train[1000]))   #to print the 1000th image in our X_train
# print(y_train[1000])    #to print label of our 1000th image; 9 is truck


# #VISUALISING THE DATASET
# # to create a 15x15 grid of images and their labels
# W_grid = 15
# L_grid = 15
# fig,axes=plt.subplots(L_grid,W_grid,figsize=(25,25))
# axes=axes.ravel()   #ravel is used to flatten the matrix ; ie in this case 15x15 matrix to 225 arrays.
# n_training=len(X_train)
# for i in np.arange(0,L_grid*W_grid):
#     index=np.random.randint(0,n_training)    #pick a random nmber
#     axes[i].imshow(X_train[index])    #store a random image (from 0 to 50000) in that index which ranges from 0 to 225.
#     axes[i].set_title(y_train[i])   #displays the label of that image as the title of corresponding image
#     axes[i].axis('off')   #removes axes numbering
# plt.subplots_adjust(hspace=0.4)


#DATA PREPARATION
X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

import keras
y_train=keras.utils.to_categorical(y_train, 10)   #to convert y_train and test values to categorical data; 10 is used as we have 10 categories
y_test=keras.utils.to_categorical(y_test, 10)

#perform data normalization
X_train=X_train/255    #this is why we convert X_train and X_test to float as this allows decimal values to be contained
X_test=X_test/255

Input_shape=X_train.shape[1:]    #gives output (32,32,3)



