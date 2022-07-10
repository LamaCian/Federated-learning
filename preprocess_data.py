from cProfile import label
import tensorflow as tf
import tensorflow_federated as tff
import pandas as pd
from create_clientdata import create_clientdata
from make_fed_data import make_federated_data


def turn_data_to_fed(dataset_name, train_set, test_set, base_model):

    print("-------- starts preprocessing --------")

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

    elif dataset_name == "Covid":
        # add labels for covid
        labels = ["COVID", "LUNG_OPACITY", "PNEUMONIA"]
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

    print("\n-------- creating ClientData Object --------")

    client_data_train, client_data_test = create_clientdata(
        client_ids, dataset_name, train_set, test_set, labels, base_model
    )
    print("\nStructure of client data train: ", client_data_train.element_type_structure)

    print("\nStructure of client data train: ", client_data_test.element_type_structure)


    print("\n-------- turn data into fed data --------")
    fed_data_test = make_federated_data(client_data_test, client_ids)

    print("\n-------- data is preprocessed --------")

    return fed_data_test, client_data_train
