from pathlib import Path
import shutil
import pandas as pd
import tensorflow as tf
import os
import tensorflow_federated as tff
import numpy as np
import nest_asyncio

nest_asyncio.apply()


def load_model(name, base_model):

    # input_shape = 96

    if name == "Leukemia":
        num_classes = 15
        input_shape = 224

        client_ids = ["0", "1", "2"]
    elif name == "Covid":
        client_ids = [
            "15",
            "19",
            "14",
            "7",
            "12",
            "16",
            "9",
            "10",
            "8",
            "0",
            "6",
            "13",
            "4",
            "3",
            "18",
        ]
        num_classes = 3
        input_shape = 96

    """Loads pre-trained model

    Args:
        base_model (keras.model): pre-trained model
        input_shape (_type_): _description_
        num_classes (_type_): _description_

    Returns:
        keras.model: Pre-trained model ready for transfer learning
    """
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

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # x = keras.layers.Dropout(0.2)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    return model
    # change input_spec -> use client_data


# files_path = Path("/home/ubuntu/federated_mentorship/test/test/files/0/")

# list_ds = tf.data.Dataset.list_files(str(files_path))
# for ds in list_ds:
#     print(ds)
# print(len(list_ds))
client_ids = [
    "15",
    "19",
    "14",
    "7",
    "12",
    "16",
    "9",
    "10",
    "8",
    "0",
    "6",
    "13",
    "4",
    "3",
    "18",
]
labels = ["COVID", "LUNG_OPACITY", "PNEUMONIA"]
labels_tf = tf.convert_to_tensor(labels)


def parse_image(filename):
    parts = tf.strings.split(filename, sep="_")
    label_str = parts[3]
    label_int = tf.where(labels_tf == label_str)[0][0]

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [96, 96])
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, label_int


NUM_EPOCHS = 5
BATCH_SIZE = 20  # faster training bigger batch size, >100 if 32x32,
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess_fl(dataset):

    return (
        dataset.repeat(NUM_EPOCHS)
        .shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER)
    )


data = []
for client in client_ids:
    files_path = Path("/home/ubuntu/federated_mentorship/test/test/files/" f"{client}/")
    print(client)
    list_ds = tf.data.Dataset.list_files(str(files_path/"*"))

    images_ds = preprocess_fl(list_ds.map(parse_image))
    # print(images_ds)
    print(list_ds)
    data.append(images_ds)

print(images_ds)
print(len(images_ds))

# print(labels_tf)
# files_path = Path("/home/ubuntu/federated_mentorship/test/test/files/0/")
# for item in files_path.glob("*"):
#     print(item.name)
#     parts = tf.strings.split(str(item), sep="_")[3]
#     label_int = tf.where(labels_tf == parts)

#     print("parts: ", label_int)
# print(len(data))

input_spec = data[0]
# print(input_spec)


def create_FL_model():

    """create_FL_model_test _summary_

    Returns:
        tff.learning.Model: _description_
    """
    keras_model = load_model("Covid", "ResNet")
    # keras_model.save_weights("model_weights.hdf5")

    return tff.learning.from_keras_model(
        keras_model,
        input_spec=input_spec.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


# for item in images_ds:
#     print(item)
#     print(len(list_ds))
# len(images_ds)
# print(data)
# for item in files_path.glob("*"):
#     print(item.name)
#     parts = tf.strings.split(item.name, sep="_")[2]
#     print("parts: ", parts)


# new_file_path = next(iter(list_ds))
# image, label = parse_image(new_file_path)
# print(label.numpy())

# example_dataset = client_data.create_tf_dataset_for_client(client_data.client_ids[0])


# def create_keras_model():
#     return tf.keras.models.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(128, 128, 3)),
#             tf.keras.layers.Dense(3, kernel_initializer="zeros"),
#             tf.keras.layers.Softmax(),
#         ]
#     )


# def model_fn():
#     # We _must_ create a new model here, and _not_ capture it from an external
#     # scope. TFF will call this within different graph contexts.
#     keras_model = create_keras_model()
#     return tff.learning.from_keras_model(
#         keras_model,
#         input_spec=input_spec.element_spec,
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     )


iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
    create_FL_model,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)
state = iterative_process.initialize()

state, metrics = iterative_process.next(state, data)
