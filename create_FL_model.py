from load_pretrained_model import load_pretrained_model
import tensorflow as tf
import tensorflow_federated as tff
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
import argparse


def create_FL_model():

    """create_FL_model_test _summary_

    Returns:
        tff.learning.Model: _description_
    """
    keras_model = load_pretrained_model(base_model, input_shape, num_classes)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
