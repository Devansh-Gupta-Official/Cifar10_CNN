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

#BUILIDNG THE MODEL
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=Input_shape))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.3))   #to remove neurons and perform regularisation

cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Dropout(0.2))   #to remove neurons and perform regularisation

cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=512, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=10, activation='softmax'))  #10 units as we need 10 outputs; softmax activation fn is used as our output has to be only 0s or 1s.


#TRAINING THE CNN
cnn.compile(optimizer = keras.optimizers.RMSprop(lr=0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

#capture the progression of the model as it is being trained
cnn.fit(X_train,y_train, batch_size=32, epochs = 2, shuffle=True)  #shuffle introduces randomness which is impoRtant to stop the model from overfitting


#EVALUATING OUR MODEL
print('Test Accuracy: {}'.format(cnn.evaluate(X_test,y_test)[1]))   #1 as evaluation comes back in 2 parts and we need the second part

predicted_classes = np.argmax(cnn.predict(X_test), axis=-1)
print(predicted_classes)

#comparing predicted value to y_test
y_test=y_test.argmax(1)   #returns original y_test which has not been converted to categorical data


#making a matrix to compare y_test and predicted_classes
L=7
W=7
fig,axes=plt.subplots(L,W,figsize=(12,12))
axes=axes.ravel()
for i in np.arange(0,L*W):
    axes[i].imshow(X_test[i])
    axes[i].set_title('Prediction={}\n True={}'.format(predicted_classes[i],y_test[i]))
    axes[i].axis('off')
plt.subplots_adjust(wspace=1)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm=confusion_matrix(y_test,predicted_classes)
print(cm)
plt.figure(figsize=(10,10))
sns.heatmap(cm,annot=True)    #presents confusion matrix in a better way


#SAVING THE MODEL
import os
directory = os.path.join(os.getcwd(),'saved model')    #to get current working directory
if not os.path.isdir(directory):        #if there is no directory called 'saved model'
    os.makedirs(directory)
model_path = os.path.join(directory,'cifar10_cnn.h5')
cnn.save(model_path)


