import numpy as np

from generate_csv import *
from dirichlet import *


def split_data(folder_path, make_csv=False):

    """Splits data into train and test set, uses dirichlets distribution on train set
    in order to simulate non-iid data distributions

    Args:
        folder_path (path/str): Path of a dataset that should be split
        make_csv (bool, optional): Generate a csv file. Defaults to False.

    Returns:
        list: train, test data
    """

    test_data = []
    train_img_paths_list = []
    train_data = []

    if folder_path == "Leukemia":

        n_parties = 3
        beta = 0.88
        num_clients = 4

        data_root_dir = Path("Leukemia")

        # iterate over all sub direcotries
        for sub_dir in data_root_dir.iterdir():
            if sub_dir.is_dir():
                # get label from name of the current directory e.g. PMO
                label = sub_dir.name

                # iterate over all images in sub directory
            for i, image_path in enumerate(sub_dir.glob("*.*")):

                # assign the image to a client

                client_id = i % num_clients

                if client_id == 3:
                    test_data.append([str(image_path.resolve()), label])

                else:
                    train_data.append([str(image_path.resolve()), label, client_id])
                    train_img_paths_list.append(str(image_path.resolve()))

        batch_idxs = dirichlet(n_parties, len(train_img_paths_list), beta)

        if make_csv == True:

            generate_csv(
                folder_path, train_data, test_data, train_img_paths_list, batch_idxs
            )
            

    return train_data, test_data
