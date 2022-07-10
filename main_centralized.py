import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import os
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


df_train = pd.read_csv("covid_without_normal_train.csv")
# df_train = pd.read_csv("covid_without_normal_train.csv")
# df_test =

test_data = pd.read_csv("covid_without_normal_test.csv")

validation = df_train[df_train["client_id"] == 14]

train_data = df_train.drop(df_train[df_train.client_id == 14].index)

# x_train = train_data["path"]
y_train = train_data["label"]


# x_test = test_data["path"]
# y_test = test_data["label"]

number_of_classes = len(np.unique(y_train))
print(number_of_classes)

# labels = [
#     "KSC",
#     "MYO",
#     "NGB",
#     "MON",
#     "PMO",
#     "MMZ",
#     "EBO",
#     "MYB",
#     "NGS",
#     "BAS",
#     "MOB",
#     "LYA",
#     "LYT",
#     "EOS",
#     "PMB",
# ]
labels = ["COVID", "LUNG_OPACITY", "PNEUMONIA"]

# ENCODING
# labelencoder = LabelEncoder()

# y_train = labelencoder.fit_transform(y_train)
# y_train = tf.keras.utils.to_categorical(y_train, number_of_classes)

# y_test = labelencoder.transform(y_test)
# y_test = tf.keras.utils.to_categorical(y_test, number_of_classes)

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
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [96, 96])

    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, label_int


path = train_data["path"]
label = train_data["label"]

path_val = validation["path"]
label_val = validation["label"]

dataset_val = tf.data.Dataset.from_tensor_slices((path_val, label_val))
dataset_val.shuffle(len(path_val))
dataset_val = dataset_val.map(parse_function, num_parallel_calls=4)
dataset_val = dataset_val.batch(batch_size=20)
dataset_val = dataset_val.prefetch(1)


path_test = test_data["path"]
label_test = test_data["label"]
dataset_test = tf.data.Dataset.from_tensor_slices((path_test, label_test))
dataset_test.shuffle(len(path_test))
dataset_test = dataset_test.map(parse_function, num_parallel_calls=4)
dataset_test = dataset_test.batch(batch_size=20)
dataset_test = dataset_test.prefetch(1)
# Create the dictionary to create a clientData
# columns_names = df_train.columns.values[1:]

# Takes a dictionary with train, validation an test sets and the desired set type


dataset = tf.data.Dataset.from_tensor_slices((path, label))
print(dataset)

dataset = dataset.shuffle(len(path))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.batch(batch_size=20)
dataset = dataset.prefetch(1)

print(dataset)
# data = tff.simulation.datasets.TestClientData(train_data)

input_shape = 96

base_model = tf.keras.applications.resnet.ResNet50(
    include_top=False,
    weights="imagenet",
    input_tensor=tf.keras.layers.Input(shape=(input_shape, input_shape, 3)),
    # input_shape=(32, 32, 3),
    pooling=None,
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(input_shape, input_shape, 3))
x = base_model(inputs, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)
# take the same validation split and create ds from that

model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
)
checkpoint_filepath = "/home/ubuntu/federated_mentorship/test/test/"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
)

history = model.fit(
    dataset,
    epochs=30,
    callbacks=[model_checkpoint_callback],
    validation_data=dataset_val,
)


epochs = history.epoch
loss = history.history["loss"]
validation_loss = history.history["val_loss"]
plt.figure(1)
plt.plot(epochs, loss, label="training")
plt.plot(epochs, validation_loss, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.savefig("covid_loss.png", bbox_inches="tight")

acc = history.history["sparse_categorical_accuracy"]
validation_acc = history.history["val_sparse_categorical_accuracy"]
plt.figure(2)
plt.plot(epochs, acc, label="training")
plt.plot(epochs, validation_acc, label="validation")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.savefig("covid_accuracy.png", bbox_inches="tight")

test_scores = model.evaluate(dataset_test)

print("\nTest: ", test_scores)


# Fine tunning

# base_model.trainable = True

# # It's important to recompile your model after you make any changes
# # to the `trainable` attribute of any inner layer, so that your changes
# # are take into account
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),  # Very low learning rate
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     metrics=[
#         tf.keras.metrics.SparseCategoricalAccuracy(),
#     ],
# )

# # Train end-to-end. Be careful to stop before you overfit!
# history = model.fit(dataset, epochs=1)
