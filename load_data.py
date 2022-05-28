import argparse
import pandas as pd

# {"path": "/home", "category": "BLA", "image": numpy.ndary"}


def load(name):

    """Loads choosen dataset

    Returns:
        _type_: Path/folder of choosen dataset
    """

    if name == "Leukemia":
        print("L")
        # file_path = path('leukemia')
        train_data = pd.read_csv("AML_train_dirichlet.csv")
        test_data = pd.read_csv("AML_test.csv")
    elif name == "Covid":
        print("C")
        # file_path = path('covid')
    # return file_path
    # eturn file_path
    return train_data, test_data


# if __name__ == "__main__":

#     parser = argparse.ArgumentParser()

#     parser.add_argument(
#         "--name",
#         help="Name of dataset you want to select",
#         type=str,
#         choices=["Leukemia", "Covid", "Breast Cancer"],
#     )

#     args = parser.parse_args()
#     name = args.name

# train, test = load(name)
# print(train)
