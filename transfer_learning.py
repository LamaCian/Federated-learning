import random
from requests import head
import tensorflow as tf
import tensorflow_federated as tff
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from make_fed_data import make_federated_data
import pandas as pd
import time
import datetime
from pathlib import Path
import pathlib
import csv
from preprocess_func import preprocess


def date_to_str(date):
    for separator in date:
        if separator == "/":
            date = date.replace(separator, "_")
    return date


def transfer_learning(name, base_model, fed_alg, client_data, num_rounds):
    if name == "Leukemia":
        input_shape = 32
        num_classes = 15
        client_ids = ["0", "1", "2"]
    input_spec = preprocess(
        client_data.create_tf_dataset_for_client(client_data.client_ids[0])
    )

    def load_model(name, base_model):
        print("loading base_model")

        if name == "Leukemia":
            input_shape = 32
            num_classes = 15
            client_ids = ["0", "1", "2"]

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

        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs, outputs)

        return model

    # change input_spec -> use client_data
    def create_FL_model():

        """create_FL_model_test _summary_

        Returns:
            tff.learning.Model: _description_
        """
        keras_model = load_model(name, base_model)
        keras_model.save_weights("model_weights.hdf5")

        # if one_hot == 0:
        #     metrics = [
        #         tf.keras.metrics.SparseCategoricalAccuracy(),
        #         tf.keras.metrics.SparseCategoricalCrossentropy(),
        #     ]
        # else:
        #     metrics = (
        #         [
        #             tf.keras.metrics.CategoricalAccuacy(),
        #             tf.keras.metrics.CategoricalCrossentropy(),
        #         ],
        #     )

        return tff.learning.from_keras_model(
            keras_model,
            input_spec=input_spec.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(),
                tf.keras.metrics.SparseCategoricalCrossentropy(),
            ],
        )

    # check again
    if fed_alg == "FedAvg":

        transfer_learning_iterative_process = (
            tff.learning.build_federated_averaging_process(
                create_FL_model,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(
                    learning_rate=0.02
                ),  # 0.2
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
            )
        )

    # elif fed_alg == "FedProx":
    #     ...

    state = transfer_learning_iterative_process.initialize()
    date_and_time = datetime.datetime.now()
    date_temp = date_and_time.strftime("%x")
    date = date_to_str(date_temp)
    num_epochs = num_rounds
    start = time.time()
    training_round = []
    training_metrics = []
    list_random_clients_ids = []
    training_info_dict = {}
    list_info = []
    list_loss = []
    list_sparse_categorical_accuracy = []
    list_sparse_categorical_crossentropy = []
    list_num_examples = []
    list_num_batches = []

    for epoch in range(num_epochs):

        start_train = time.time()

        print("training")

        subfolder_path = Path("output/" f"{date}_{base_model}_{fed_alg}")
        subfolder_path.mkdir(parents=True, exist_ok=True)
        title = "state"
        (subfolder_path / f"{title}.txt").write_text(str(state.model))
        file_path = subfolder_path / "train_info.csv"

        random_clients_ids = random.sample(client_ids, k=2)

        federated_train_data = make_federated_data(client_data, random_clients_ids)
        state, metrics = transfer_learning_iterative_process.next(
            state, federated_train_data
        )

        train_metrics = metrics["train"]

        list_random_clients_ids.append(random_clients_ids)
        training_info = {}
        training_round.append(epoch)
        training_metrics.append(metrics)

        loss = train_metrics["loss"]
        num_examples = train_metrics["num_examples"]
        num_batches = train_metrics["num_batches"]
        train_metrics = metrics["train"]
        sparse_categorical_accuracy = train_metrics["sparse_categorical_accuracy"]
        sparse_categorical_crossentropy = train_metrics[
            "sparse_categorical_crossentropy"
        ]

        list_sparse_categorical_accuracy.append(
            train_metrics["sparse_categorical_accuracy"]
        )
        list_sparse_categorical_crossentropy.append(
            train_metrics["sparse_categorical_crossentropy"]
        )
        list_loss.append(train_metrics["loss"])
        list_num_examples.append(train_metrics["num_examples"])
        list_num_batches.append(train_metrics["num_batches"])

        end = time.time()

        train_time = end - start_train

        total_time = end - start

        if (train_time / 60) > 60:
            print("Time spent on training: ~ {:.2f} in hours".format(train_time / 3600))
        else:
            print("Time spent on training: {:.2f} in minutes".format(train_time / 60))

        print(
            "loss: {}, sparse_accuracy: {}, sparse_categorical_crossentropy : {}".format(
                train_metrics["loss"],
                train_metrics["sparse_categorical_accuracy"],
                train_metrics["sparse_categorical_crossentropy"],
            )
        )

        training_info = pd.DataFrame(
            {
                "selected clients id": list_random_clients_ids,
                "sparse_categorical_accuracy": list_sparse_categorical_accuracy,
                "sparse_categorical_crossentropy": list_sparse_categorical_crossentropy,
                "loss": list_loss,
                "num_examples": list_num_examples,
                "num_batches": list_num_batches,
            }
        )
        list_info.append(
            [
                sparse_categorical_accuracy,
                sparse_categorical_crossentropy,
                loss,
                num_examples,
                num_batches,
                total_time,
            ]
        )

        training_info.to_csv(file_path, index=False)

    print("Total time spent on training: {:.2f} in min".format(total_time / 60))

    # print("all cool")

    return state
