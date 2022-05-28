import tensorflow as tf
import tensorflow_federated as tff
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
import argparse
from transfer_learning import *


def load_model(base_model, input_shape, num_classes):

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

    outputs = tf.keras.layers.Dense(num_classes)(x)
    model = tf.keras.Model(inputs, outputs)

    return model


def create_FL_model_test():
    """create_FL_model_test _summary_

    Returns:
        _type_: _description_
    """
    keras_model = load_model(base_model, input_shape=32, num_classes=15)
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=preprocessed_example_dataset.element_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )


transfer_learning_iterative_process = tff.learning.build_federated_averaging_process(
    create_FL_model_test,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
)

state = transfer_learning_iterative_process.initialize()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        help="Which pre-trained model you want to select",
        type=str,
        choices=["VGG16", "ResNet", "EfficientNet"],
    )
    parser.add_argument(
        "--input_shape",
        help="Which pre-trained model you want to select",
        type=str,
    )

    args = parser.parse_args()
    base_model = args.base_model
    input_shape = args.input_shape

    load_model(
        base_model,
    )
