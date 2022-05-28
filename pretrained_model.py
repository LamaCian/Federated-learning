import tensorflow as tf
import tensorflow_federated as tff
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization


def load_model(name, base_model):

    if name == "Leukemia":
        input_shape = 32
        num_classes = 15

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
