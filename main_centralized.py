import pandas as pd
import tensorflow as tf
import os
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import keras
from sklearn.preprocessing import LabelEncoder
from preprocess_func import preprocess

train_data = pd.read_csv("AML_train_dirichlet.csv")
test_data = pd.read_csv("AML_test.csv")


x_train = train_data["path"]
y_train = train_data["label"]


x_test = test_data["path"]
y_test = test_data["label"]

number_of_classes = len(np.unique(y_train))
print(number_of_classes)

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

# ENCODING
labelencoder = LabelEncoder()

y_train = labelencoder.fit_transform(y_train)
y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)

y_test = labelencoder.transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)

# preprocesiranje

# ucitavanje base_model i tu treba da izvrsim transfer learning
# metrike

labels_tf = tf.convert_to_tensor(labels)


def parse_function(filename, label):

    parts = tf.strings.split(filename, os.sep)
    label_str = parts[-2]

    label_int = tf.where(labels_tf == label_str)[0][0]
    image_string = tf.io.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, label_int


path = train_data["path"]
label = train_data["label"]

dataset = tf.data.Dataset.from_tensor_slices((path, label))
print(dataset)

dataset = dataset.shuffle(len(path))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.batch(batch_size=20)
dataset = dataset.prefetch(1)

print(dataset)


input_shape = 32

base_model = tf.keras.applications.resnet.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=(32, 32, 3)),
    # input_shape=(32, 32, 3),
    pooling=None,
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(32, 32, 3))
x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)


model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
        tf.keras.metrics.SparseCategoricalCrossentropy(),
    ],
)

model.fit(dataset, epochs=5)


base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are take into account
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
)

# Train end-to-end. Be careful to stop before you overfit!
history = model.fit(dataset, epochs=1)
