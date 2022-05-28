import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
from create_clientdata import *


def turn_data_to_fed(dataset_name):

    """Preprocess the input data

    Args:
        dataset_name (str): name of choosen dataset

    Returns:
        _type_: Preprocessed and federated dataset
    """

    if dataset_name == "Leukemia":
        labels = [
            "KSC",
            "MYO",
            "NGB",
            "MON",
            "PMO",
            "MMZ",
            "EBO",
            "MYB",
            "NGS",
            "BAS",
            "MOB",
            "LYA",
            "LYT",
            "EOS",
            "PMB",
        ]
        client_ids = ["0", "1", "2"]

    elif dataset_name == "Covid":
        # add labels for covid
        labels_tf = tf.convert_to_tensor(labels)

    client_data = create_clientdata(client_ids)

    # def parse_image(filename):

    #     parts = tf.strings.split(filename, os.sep)
    #     label_str = parts[-2]

    #     label_int = tf.where(labels_tf == label_str)[0][0]
    #     image = tf.io.read_file(filename)
    #     image = tf.io.decode_jpeg(image, channels=3)  #
    #     image = tf.image.convert_image_dtype(image, tf.float32)

    #     return image, label_int

    # def create_dataset(client_id):
    #     df = pd.read_csv("AML_train_dirichlet.csv", encoding="utf-8")

    #     client_id = int(client_id)

    #     file = df.loc[df["client_id"] == client_id]
    #     # print(file)
    #     path = file["path"]

    #     # print(path)
    #     list_ds = tf.data.Dataset.list_files(path)

    #     images_ds = list_ds.map(parse_image)

    #     return images_ds

    # client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
    #     client_ids, create_dataset
    # )

    NUM_EPOCHS = 5
    BATCH_SIZE = 20
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10

    def preprocess(dataset):

        return (
            dataset.repeat(NUM_EPOCHS)
            .shuffle(SHUFFLE_BUFFER, seed=1)
            .batch(BATCH_SIZE)
            .prefetch(PREFETCH_BUFFER)
        )

    def make_federated_data(client_data, client_ids):

        """Construct a list of datasets from the given set of users, which will be later used as an input
        to a round of training or evaluation.

        Args:
            client_data (TensorSpec/ClientData object): _description_
            client_ids (list): A list of string identifiers for clients in this dataset.


        Returns:
            tf.data.Dataset: federated dataset
        """
        return [
            preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids
        ]

    return preprocessed_data
