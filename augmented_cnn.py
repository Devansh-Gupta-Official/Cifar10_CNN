#IMAGE OR DATA AUGMENTATION
#changing images(rotating, cropping, etc) and training the model on the changed or updated images to bring it outside of its comfort zone and to train it on a varied and distinct image set.
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10

(X_train,y_train), (X_test,y_test) = cifar10.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')


import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rotation_range=90)
train_datagen.fit(X_train)

#to create endless variations of our image
from PIL import Image
# from scipy.misc import toimage  #this is removed after update; use the one below instead
fig=plt.figure(figsize=(20,20))
n=8
X_train_sample = X_train[:n]
for x_batch in train_datagen.flow(X_train_sample, batch_size=n):
    for i in range(0,n):
        ax=fig.add_subplot(1,n,i+1)
        ax.imshow(Image.fromarray(x_batch[i]*255).astype(np.uint8).resize(20,20).convert('RGB'))

    fig.suptitle('Augmented images')
    plt.show()
    break;