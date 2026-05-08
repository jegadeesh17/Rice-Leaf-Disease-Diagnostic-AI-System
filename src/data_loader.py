import os
import tensorflow as tf
import keras
from keras import layers, Sequential

def get_augmentation_layer():
    """Returns a sequential layer for image augmentation."""
    return Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ], name="data_augmentation")

def load_datasets(train_path, val_path, test_path, img_size=224, batch_size=8):
    """Loads and optimizes training, validation, and test datasets."""
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_path, image_size=(img_size, img_size), batch_size=batch_size, label_mode='categorical'
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_path, image_size=(img_size, img_size), batch_size=batch_size, label_mode='categorical'
    )
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_path, image_size=(img_size, img_size), batch_size=batch_size, label_mode='categorical', shuffle=False
    )

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds
