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


def transfer_learning(
    name, base_model, fed_alg, client_data, num_rounds, learning_manner
):
    input_shape = 96

    input_spec = preprocess(
        client_data.create_tf_dataset_for_client(client_data.client_ids[0])
    )
    if name == "Leukemia":
        num_classes = 15
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

    def load_model(name, base_model):

        input_shape = 96

        if name == "Leukemia":
            num_classes = 15

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
    def create_FL_model():

        """create_FL_model_test _summary_

        Returns:
            tff.learning.Model: _description_
        """
        keras_model = load_model(name, base_model)
        keras_model.save_weights("model_weights.hdf5")

        return tff.learning.from_keras_model(
            keras_model,
            input_spec=input_spec.element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

    # if fed_alg == "FedAvg":

    #     transfer_learning_iterative_process = (
    #         tff.learning.algorithms.build_unweighted_fed_prox(
    #             create_FL_model,
    #             #
    #             server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    #         )
    #     )

    # # check again
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

    state_transfer = transfer_learning_iterative_process.initialize()

    keras_model = load_model(name="Covid", base_model="ResNet")

    state = tff.learning.state_with_new_model_weights(
        state_transfer,
        trainable_weights=[v.numpy() for v in keras_model.trainable_weights],
        non_trainable_weights=[v.numpy() for v in keras_model.non_trainable_weights],
    )

    def keras_train(state, round_num):

        keras_model = load_model(name="Covid", base_model="ResNet")
        keras_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.sparse_categorical_accuracy],
        )
        state.model.assign_weights_to(keras_model)
        model_weights = transfer_learning_iterative_process.get_model_weights(state)
        model_weights.assign_weights_to(keras_model)
        loss, accuracy = keras_model.evaluate(fed_valid_data, steps=2, verbose=0)
        print("\tEval: loss={l:.3f}, accuracy={a:.3f}".format(l=loss, a=accuracy))

    date_and_time = datetime.datetime.now()
    date_temp = date_and_time.strftime("%x")
    date = date_to_str(date_temp)
    num_epochs = 2
    start = time.time()
    training_round = []
    training_metrics = []
    list_random_clients_ids = []
    training_info_dict = {}
    all_info = []
    loss = []
    sparse_categorical_accuracy = []
    num_examples = []
    num_batches = []
    list_valid_ids = []
    eval_sparse_categorical_accuracy = []
    eval_loss = []

    # client_data_train, client_data_valid = client_data.train_test_client_split(
    #     client_data, num_test_clients=1, seed=12345
    # )
    # print(client_data_train.client_ids)
    # client_train_ids = client_data_train.client_ids
    # print(client_data_valid.client_ids)
    federated_eval = tff.learning.build_federated_evaluation(create_FL_model)

    for round in range(num_rounds):

        start_train = time.time()

        subfolder_path = Path("output/" f"{date}_{base_model}_{fed_alg}")
        subfolder_path.mkdir(parents=True, exist_ok=True)

        file_path = subfolder_path / "train_info.csv"

        client_data_train, client_data_valid = client_data.train_test_client_split(
            client_data, num_test_clients=1, seed=1234
        )
        client_train_ids = client_data_train.client_ids

        random_clients_ids = random.sample(client_train_ids, k=2)

        federated_train_data = make_federated_data(
            client_data_train, random_clients_ids
        )
        valid_ids = client_data_valid.client_ids

        fed_valid_data = preprocess(
            client_data_valid.create_tf_dataset_for_client(
                client_data_valid.client_ids[0]
            )
        )
        # print("-------- training starts --------")
        # state, metrics = transfer_learning_iterative_process.next(
        #     state, federated_train_data
        # )

        keras_train(state, round)
        state, metrics = transfer_learning_iterative_process.next(
            state, federated_train_data
        )

        list_random_clients_ids.append(random_clients_ids)
        training_info = {}
        training_round.append(round)
        training_metrics.append(metrics)
        list_valid_ids.append(valid_ids)

        train_metrics = metrics["train"]
        print(train_metrics)

        loss.append(train_metrics["loss"])
        num_examples.append(train_metrics["num_examples"])
        sparse_categorical_accuracy.append(train_metrics["sparse_categorical_accuracy"])
        num_batches.append(train_metrics["num_batches"])

        end = time.time()

        train_time = end - start_train

        total_time = end - start

        if (train_time / 60) > 60:
            print("Time spent on training: ~ {:.2f} in hours".format(train_time / 3600))
        else:
            print("Time spent on training: {:.2f} in minutes".format(train_time / 60))

        print(
            "loss: {}, sparse_accuracy: {}".format(
                train_metrics["loss"],
                train_metrics["sparse_categorical_accuracy"],
            )
        )

        # training_info = pd.DataFrame({"selected ids": list_random_clients_ids})
        # # 'sparse_categorical_accuracy': sparse_categorical_accuracy})

        # all_info.append(
        #     [
        #         sparse_categorical_accuracy,
        #         loss,
        #         num_examples,
        #         num_batches,
        #         total_time,
        #     ]
        # )

        # # print(training_info)

        # header = [
        #     "sparse_categorical_accuracy",
        #     "loss",
        #     "num_examples",
        #     "num_batches",
        #     "time",
        # ]

        # VALIDATION STEP
        #     client_data_valid

        model_weights = transfer_learning_iterative_process.get_model_weights(state)
        eval_metric = federated_eval(model_weights, [fed_valid_data])
        print(eval_metric)
        eval_sparse_categorical_accuracy.append(
            eval_metric["eval"]["sparse_categorical_accuracy"]
        )
        eval_loss.append(eval_metric["eval"]["loss"])

        training_info = pd.DataFrame(
            {
                "selected clients id": list_random_clients_ids,
                "sparse_categorical_accuracy": sparse_categorical_accuracy,
                "loss": loss,
                "num_examples": num_examples,
                "num_batches": num_batches,
                "eval_sparse_categorical_accuracy": eval_sparse_categorical_accuracy,
                "eval_loss": eval_loss,
                "valid_ids": list_valid_ids,
            }
        )

        training_info.to_csv(file_path)

        # training_info.to_csv(file_path, index=False)

    print("Total time spent on training: {:.2f} in min".format(total_time / 60))

    return state
