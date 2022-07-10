from pathlib import Path
import shutil
import pandas as pd
import tensorflow as tf
import os
import tensorflow_federated as tff
from preprocess_func import preprocess
import numpy

tf.compat.v1.disable_eager_execution()

# import nest_asyncio

# nest_asyncio.apply()


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


df = pd.read_csv("covid_without_normal_train.csv", encoding="utf-8")
result = "1"
# filepath = Path("/Users/admin/Desktop/covid/test.txt")
# for client_id in client_ids:

#     #
#     # Copy the contents of the file named src to a file named dst.
#     #  Both src and dst need to be the entire filename of the files, including path.

#     subfolder_path = Path(f"files/{client_id}/")

#     subfolder_path.mkdir(parents=True, exist_ok=True)

#     df_filtered = df[df["client_id"] == int(client_id)]
#     file_paths = df_filtered["path"]
#     for path in file_paths:
#         path_obj = Path(path)
#         file_name = path_obj.name
#         # path_name = path_str[38:].replace("/", "_")
#         new_path = subfolder_path / file_name
#         shutil.copy(path, new_path)

# with subfolder_path.open("w", encoding ="utf-8") as f:

#     f.write(result)
# labels = ["COVID", "LUNG_OPACITY", "PNEUMONIA"]
# labels_tf = tf.convert_to_tensor(labels)
# filename = (
#     "/home/ubuntu/federated_mentorship/test/test/files/3/d_raw_COVID_000001-17.jpg"
# )
# parts = tf.strings.split(filename, sep="/")
# parts_temp = parts[-1]
# label_str = tf.strings.split(parts_temp, sep="_")[2]
# label_int = tf.where(labels_tf == label_str)[0][0]

# print(parts)
# print(parts_temp)
# print("label_str", label_str)
# print("label_int", label_int)

labels = ["COVID", "LUNG_OPACITY", "PNEUMONIA"]

labels_tf = tf.convert_to_tensor(labels)


def parse_image(filename):
    # parts = tf.strings.split(filename, os.sep)
    # label_str = parts[-2]

    # label_int = tf.where(labels_tf == label_str)[0][0]

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [32, 32])  #
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image, tf.constant([1])


data = []
NUM_CLIENTS = 10
NUM_EPOCHS = 5
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess_new(dataset):

    return (
        dataset.shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER)
    )


def create_clientdata(client_id):

    files_path = Path(f"files/{client_id}/")

    list_ds = tf.data.Dataset.list_files(str(files_path) + "/*/*")
    images_ds = list_ds.map(parse_image)
    # data.append(images_ds)

    return images_ds


client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    client_ids, create_clientdata
)
print(client_data.client_ids)

example_dataset = client_data.create_tf_dataset_for_client(client_data.client_ids[0])
# print(len(example_dataset))
# for example in example_dataset:
#     print(example)

centralized = client_data.create_tf_dataset_from_all_clients(seed=7)

# for x in centralized.take(5):
#     print(x)

pre_centralized = preprocess_new(centralized)
print(pre_centralized)
# for example in example_dataset:
#     print(example)
input_shape = 32

number_of_classes = 3

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


model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(),
    ],
)


model.fit(pre_centralized, epochs=40)
# def make_federated_data(images_ds, client_ids):
#     return [preprocess(images_ds.create_tf_dataset_for_client(x)) for x in client_ids]


# fed_data = make_federated_data(client_data, client_ids)

# print(fed_data)

# input_spec = fed_data[0]
# print(input_spec)


# def create_keras_model():
#     return tf.keras.models.Sequential(
#         [
#             tf.keras.layers.InputLayer(input_shape=(784,)),
#             tf.keras.layers.Dense(10, kernel_initializer="zeros"),
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


# iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
#     model_fn,
#     client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
#     server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
# )
# state = iterative_process.initialize()

# state, metrics = iterative_process.next(state, fed_data)


# client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
#     client_ids, create_clientdata
# )

# print(client_data.client_ids)

# centralized = client_data.create_tf_dataset_from_all_clients()

# example_dataset = client_data.create_tf_dataset_for_client(client_data.client_ids[1])

# for example in example_dataset:
#     print(example)
