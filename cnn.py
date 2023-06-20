import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt

from keras.datasets import cifar10

(X_train,y_train), (X_test,y_test) = cifar10.load_data()
print(X_train.shape)   #50000 samples and each image has 32x32 dimensions and rgb or 3 channels
print(y_train.shape)   #50000 samples and each sample has 1 corresponding claasification like horse, cat, etc...

