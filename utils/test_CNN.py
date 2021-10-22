import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import tensorflow_io as tfio

# import joblib

import pathlib

def build_model():
    dataset_dir = "../data/train3"
    data_dir = pathlib.Path(dataset_dir)
    print(data_dir)
    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(image_count)

    batch_size = 32
    img_height = 256
    img_width = 256

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    print(train_ds)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # normalization_layer = layers.Rescaling(1./255)

    num_classes = 2

    data_augmentation = Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           1)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        #data_augmentation,
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
        layers.Conv2D(16, 8, padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 8, padding='same', activation='relu'),
        layers.Dropout(0.5),
        layers.MaxPooling2D(),
        # layers.Conv2D(64, 3, padding='same', activation='relu'),
        # layers.MaxPooling2D(),


        layers.Flatten(),
        # layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='relu')
    ])


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # UPD: that was only true for a small original dataset
    # I train with 15 epochs. Takes 4 sec for an epoch => 1 min total time
    # but five epochs are also fine for testing.
    epochs = 10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save('models/model4')

    # you may uncomment it to see the plot of training and validation accuracy

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    return model, img_height, img_width, class_names


if __name__ == '__main__':
    build_model()
