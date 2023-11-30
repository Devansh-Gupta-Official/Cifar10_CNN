# CIFAR-10 Image Classification with CNN and Data Augmentation

This repository contains Python scripts for image classification using Convolutional Neural Networks (CNN) on the CIFAR-10 dataset. Additionally, it includes data augmentation techniques using TensorFlow's ImageDataGenerator to enhance the model's robustness.

## **Files**
### **File 1 - cnn.py**
This file focuses on:
- Loading and visualizing the CIFAR-10 dataset
- Data preparation including normalization and categorical conversion
- Building a CNN model using TensorFlow/Keras
- Training the CNN model on the CIFAR-10 dataset
- Evaluating the model's performance on the test dataset
- Visualizing predictions and creating a confusion matrix
- Saving the trained model

### **File 2 - augumented_cnn.py**
This covers:

- Loading CIFAR-10 dataset
- Implementing data augmentation using ImageDataGenerator in Keras
- Displaying augmented images for training

## **Dataset**
The CIFAR-10 dataset is a popular dataset used for image classification tasks. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. These classes include common objects like **airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks**.
To use the CIFAR-10 dataset in a Jupyter Notebook (ipynb), you'll first need to download it. You can download it directly using TensorFlow or PyTorch libraries, as they provide built-in functions for accessing popular datasets like CIFAR-10.
Here's a sample code snippet to download CIFAR-10 using TensorFlow:
```
import tensorflow as tf

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0
```


## **Usage**
1. Clone the repository to your local environment:
   ```
   git clone https://github.com/Devansh-Gupta-Official/cifar10_CNN.git
   ```
2. Open and run the file in your preferred environment (Jupyter Notebook, Google Colab, etc.).
3. File 1 (cnn.py) demonstrates the CNN model training process on the CIFAR-10 dataset and provides insights into its performance.
4. File 2 (augumented_cnn.py) showcases data augmentation techniques using TensorFlow's ImageDataGenerator, providing visual examples of augmented images.

## **Requirements**
- Python 3
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Seaborn
- PIL (Python Imaging Library)

## **Note**
- The code assumes basic familiarity with TensorFlow/Keras, CNNs, and image classification concepts.
- Adjust hyperparameters, model architecture, or augmentation techniques for experimentation and improved performance.
