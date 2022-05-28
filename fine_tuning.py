from transfer_learning import load_model
import random
import tensorflow as tf
import tensorflow_federated as tff
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from make_fed_data import make_federated_data


def fine_tuning(name, base_model, input_spec, state, client_data):
    if name == "Leukemia":
        clients_ids = ["0", "1", "2"]

    def fine_tuning_keras_model():
        model = load_model(name, base_model)

        state.model.assign_weights_to(model)

        # load model keras fn #path to weights file
        # tf.keras.Model.load_weights(model,filepath = '/Users/admin/Desktop/AML_intro/')
        # load_weights('/Users/admin/Desktop/AML_intro/checkpoint')#     model.trainable = true
        # go through all layer and set trainable to true
        model.trainable = True

        # initalize a new keras model
        #

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
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.0002),
    )

    state_fine_tuning = fine_tuning_iterative_process.initialize()
    random_clients_ids = random.sample(clients_ids, k=2)

    federated_train_data = make_federated_data(client_data, random_clients_ids)
    state_fine_tuning, metrics = fine_tuning_iterative_process.next(
        state_fine_tuning, federated_train_data
    )
