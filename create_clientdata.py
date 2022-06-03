import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
import os


def create_clientdata(client_ids, train_set, labels):
    labels_tf = tf.convert_to_tensor(labels)

    def parse_image(filename):
        parts = tf.strings.split(filename, os.sep)
        label_str = parts[-2]

        label_int = tf.where(labels_tf == label_str)[0][0]

        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [32, 32])  #

        # if one_hot == 1:
        #     label_one_hot = tf.one_hot(label_int, depth=15)
        #     return image, label_one_hot
        # else:
        #     return image, label_int

        return image, label_int

    def create_dataset(client_id):

        """create_dataset _summary_

        Args:
            client_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        df = train_set

        client_id = int(client_id)

        file = df.loc[df["client_id"] == client_id]
        # print(file)
        path = file["path"]

        # print(path)
        list_ds = tf.data.Dataset.list_files(path)

        images_ds = list_ds.map(parse_image)

        return images_ds

    client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids, create_dataset
    )

    return client_data
