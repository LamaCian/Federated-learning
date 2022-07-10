import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras.callbacks import ReduceLROnPlateau

labels = [
    "KSC",
    "MYO",
    "NGB",
    "MON",
    "PMO",
    "MMZ",
    "EBO",
    "MYB",
    "NGS",
    "BAS",
    "MOB",
    "LYA",
    "LYT",
    "EOS",
    "PMB",
]

labels_tf = tf.convert_to_tensor(labels)


def parse_image(filename):
    parts = tf.strings.split(filename, os.sep)
    label_str = parts[-2]

    label_int = tf.where(labels_tf == label_str)[0][0]
    #     label_one_hot = tf.one_hot(label_int, depth = 15)

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)  # check for grayscale
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [32, 32])  # don't resize

    # decode label
    # label_str = parts[-2]
    # encode label funct and return int

    return image, label_int


def create_dataset(client_id):
    df = pd.read_csv("AML_train_dirichlet.csv", encoding="utf-8")

    client_id = int(client_id)

    file = df.loc[df["client_id"] == client_id]
    # print(file)
    path = file["path"]

    # print(path)
    list_ds = tf.data.Dataset.list_files(path)

    images_ds = list_ds.map(parse_image)

    return images_ds

    # passing the arguments to from_clients_and_tf_fn to create ClientData


client_ids = ["0", "1", "2"]
client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids, create_dataset
)


NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):

    return (
        dataset.repeat(NUM_EPOCHS)
        .shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER)
    )


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]

    #


""" DOCSTRING"""

# VGG16


def test_function(base_model, input_shape, num_classes):

    if base_model == "VGG16":
        base_model = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            weights="imagenet",
            input_tensor=tf.keras.layers.Input(shape=(input_shape, input_shape, 3)),
            pooling=None,
        )
    elif base_model == "EfficientNet":
        base_model = tf.keras.applications.efficientnet.EfficientNetB4(
            include_top=False,
            weights="imagenet",
            input_tensor=tf.keras.layers.Input(shape=(input_shape, input_shape, 3)),
            pooling=None,
        )

    elif base_model == "ResNet":
        base_model = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights="imagenet",
            input_tensor=tf.keras.layers.Input(shape=(input_shape, input_shape, 3)),
            pooling=None,
        )

    base_model.trainable = False

    inputs = tf.keras.Input(shape=(input_shape, input_shape, 3))
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # A Dense classifier with a single unit (binary classification)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)
    #     tf.keras.Model.save_weights(model,'/Users/admin/Desktop/AML_intro/AML_weights/', save_format = 'tf')

    return model


input_spec = preprocess(
    client_data.create_tf_dataset_for_client(client_data.client_ids[0])
)


def create_FL_model_test():

    keras_model = test_function(base_model="ResNet", input_shape=32, num_classes=15)
    keras_model.save_weights("model_weights.hdf5")
    #     print(keras_model.weights)

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


test_learning_iterative_process = tff.learning.build_federated_averaging_process(
    create_FL_model_test,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)
state = test_learning_iterative_process.initialize()

import random

random_clients_ids = random.sample(client_ids, k=2)
federated_train_data = make_federated_data(client_data, random_clients_ids)
state, metrics = test_learning_iterative_process.next(state, federated_train_data)


def fine_tuning_keras_model():
    model = test_function(base_model="ResNet", input_shape=32, num_classes=15)

    #     state.model.assign_weights_to(model)
    model.load_weights("model_weights.hdf5")

    model.trainable = True

    return model


def fine_tuning_FL_model():
    model = fine_tuning_keras_model()

    return tff.learning.from_keras_model(
        model,
        input_spec=input_spec.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


fine_tuning_iterative_process = tff.learning.build_federated_averaging_process(
    fine_tuning_FL_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.01),
)

state_fine_tuning = fine_tuning_iterative_process.initialize()


random_clients_ids = random.sample(client_ids, k=2)

federated_train_data = make_federated_data(client_data, random_clients_ids)
state_fine_tuning, metrics = fine_tuning_iterative_process.next(
    state_fine_tuning, federated_train_data
)
print("all cool")
