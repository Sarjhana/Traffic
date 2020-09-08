import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images, labels = [], []
    fnames = os.listdir(data_dir)
    for filename in fnames:
        print(f'Reading file : {filename}')
        img_folder = os.path.join(data_dir, filename)
        for img_path in os.listdir(img_folder):
            img = cv2.imread(os.path.join(img_folder, img_path))
            if img is not None:
                images.append(cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)))
                labels.append(int(filename))
    
    return (images, labels)

    #raise NotImplementedError


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = tf.keras.Sequential([

        #convolution layer 1
        Conv2D(64, (3, 3), activation='relu', input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)), #input layer
        MaxPool2D((2, 2)),
        Dropout(0.2),

        #convolution layer 2
        Conv2D(128, (3, 3), activation='relu'),
        MaxPool2D((2, 2)),
        Dropout(0.1),

        Flatten(),

        #hidden layer 1 with 32 neurons
        Dense(128, activation='relu'),

        #hidden layer 2 with 64 neurons
        Dense(64, activation='relu'),

        Dropout(0.1),
        Dense(NUM_CATEGORIES, activation='sigmoid') #output layer
    ])
    
    model.compile(loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy'])

    return model

    #raise NotImplementedError


if __name__ == "__main__":
    main()
