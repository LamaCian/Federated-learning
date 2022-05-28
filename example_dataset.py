import pandas as pd
from create_clientdata import create_clientdata
from preprocess_func import preprocess


def make_example_dataset(name):
    if name == "Leukemia":
        client_ids = ["0", "1", "2"]
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
        df = pd.read_csv("AML_train_dirichlet.csv", encoding="utf-8")
        train_set = df

        client_data = create_clientdata(client_ids, train_set, labels)
        example_dataset = client_data.create_tf_dataset_for_client(
            client_data.client_ids[1]
        )
        preprocessed_example_dataset = preprocess(example_dataset)

    return preprocessed_example_dataset
