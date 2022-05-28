import tensorflow_federated as tff
import tensorflow as tf
from preprocess_func import preprocess


def make_federated_data(client_data, client_ids):

    """Construct a list of datasets from the given set of users, which will be later used as an input
    to a round of training or evaluation.

    Args:
        client_data (TensorSpec/ClientData object): _description_
        client_ids (list): A list of string identifiers for clients in this dataset.


    Returns:
        tf.data.Dataset: federated dataset
    """
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]
