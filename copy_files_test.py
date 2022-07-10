from pathlib import Path
import shutil
import pandas as pd
import tensorflow as tf

import numpy

# import nest_asyncio

# nest_asyncio.apply()

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


df = pd.read_csv("covid_without_normal_train.csv", encoding="utf-8")
# result = "1"
# filepath = Path("/Users/admin/Desktop/covid/test.txt")
for client_id in client_ids:

    #
    # Copy the contents of the file named src to a file named dst.
    #  Both src and dst need to be the entire filename of the files, including path.

    subfolder_path = Path(f"files/{client_id}/")

    subfolder_path.mkdir(parents=True, exist_ok=True)

    df_filtered = df[df["client_id"] == int(client_id)]
    file_paths = df_filtered["path"]
    for path in file_paths:
        path_obj = Path(path)
        label = path_obj.parent.name
        file_name = path_obj.name
        # path_name = path_str[38:].replace("/", "_")
        label_path = subfolder_path / label
        label_path.mkdir(exist_ok=True)
        new_path = label_path / file_name
        shutil.copy(path, new_path)
