import tensorflow as tf
import tensorflow_federated as tff


def transfer_learning_centralized(dataset_name, base_model):

    if dataset_name == "Leukemia":
        input_shape = 224
        number_of_classes = 15
    elif dataset_name == "Covid":
        input_shape = 96
        number_of_classes = 3

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

    outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)

    return model.compile(
        optimizer=tf.keras.optimizers.SGD(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
        ],
    )
