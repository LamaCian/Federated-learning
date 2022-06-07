from cProfile import label
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
from create_clientdata import create_clientdata
from make_fed_data import make_federated_data


def turn_data_to_fed(dataset_name, train_set):

    """Preprocess the input data and turns it into format compatible for federated setting

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

    # elif dataset_name == "Covid":
    #     # add labels for covid
    print('-------- creating ClientData Object --------')

    client_data = create_clientdata(client_ids, train_set, labels)
    print("Structure of client data: ", client_data.element_type_structure)

    print('-------- turn data into fed data --------')
    fed_data = make_federated_data(client_data, client_ids)

    return fed_data, client_data
