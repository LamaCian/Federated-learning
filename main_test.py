from doctest import testmod
from email.mime import base
import tensorflow_federated as tff
import tensorflow
from load_data import load
from create_clientdata import create_clientdata
from preprocess_data import turn_data_to_fed
from split_data import *
import argparse
from pathlib import Path
from example_dataset import make_example_dataset
from transfer_learning2 import transfer_learning2
from distutils.util import strtobool

# from fine_tuning import fine_tuning
import csv
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    from distutils.util import strtobool

    parser.add_argument(
        "--name",
        help="Name of dataset you want to select",
        type=str,
        choices=["Leukemia", "Covid", "Breast Cancer"],
    )
    parser.add_argument(
        "--base_model",
        help="Name of pre-trained model",
        type=str,
        choices=["ResNet", "VGG16", "EfficientNet"],
    )
    parser.add_argument(
        "--fed_alg",
        help="Name of pre-trained model",
        type=str,
        choices=["FedAvg", "FedProx"],
    )
    parser.add_argument(
        "--num_rounds",
        help="Number of training rounds",
        type=int,
    )

    parser.add_argument(
        "--learning_manner",
        help=" ",
        type=str,
        choices=["Federated", "Centralized"],
    )

    # parser.add_argument(
    #     "--one_hot",
    #     type=lambda x: bool(strtobool(x)),
    #     choices=["1", "0"],
    #     default="0",
    #     help="use one-hot encoding",
    # )

    args = parser.parse_args()
    name = args.name
    base_model = args.base_model
    fed_alg = args.fed_alg
    num_rounds = args.num_rounds
    learning_manner = args.learning_manner

    train, test = load(name)

    print("\n-------- preprocessing --------")

    fed_test, client_data_train = turn_data_to_fed(name, train, test, base_model)

    print("\n-------- transfer learning --------")

    state = transfer_learning2(name, base_model, fed_alg, client_data_train, fed_test, num_rounds, learning_manner)

    # new_state = fine_tuning(name, base_model, example_dataset, state, client_data)
