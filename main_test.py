from email.mime import base
import tensorflow_federated as tff
import tensorflow
from load_data import load
from create_clientdata import *
from preprocess_data import turn_data_to_fed
from split_data import *
import argparse
from pathlib import Path
from example_dataset import make_example_dataset
from transfer_learning import transfer_learning

# from fine_tuning import fine_tuning
import csv
import pandas as pd

# data = load("Leukemia")

# train_set, test_set = split_data("Leukemia", make_csv=True)

# fed_train_set = turn_data_to_fed("Leukemia")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

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

    args = parser.parse_args()
    name = args.name
    base_model = args.base_model
    fed_alg = args.fed_alg

    train, test = load(name)

    fed_train_set, client_data = turn_data_to_fed(name, train)

    example_dataset = make_example_dataset(name)

    state = transfer_learning(name, base_model,  fed_alg, client_data)

    # new_state = fine_tuning(name, base_model, example_dataset, state, client_data)

    # create table of loss, metrics..
    # save weights to specific folder
    # output_dir --> save weights there
    # for testing load everything from output_dir
    # save start of training ->
    # output_dir ->

    # making directory
    # import os

    # parent_dir = os.getcwd()
    # directory = "output_dir"
    # path = os.path.join(parent_dir, directory)
    # output_dir = os.mkdir(path)
    # for rounds in NUM_RUNDS:
    #     name_of_sub_directory = str(datetime) + "_" + str(model)...
    #     path_of_subdir = os.path.join(output_dir, sub_directory)
    #     sub_dir = os.mkdir(path_of_subdir)
