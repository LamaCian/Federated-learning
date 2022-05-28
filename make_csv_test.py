import glob
from pathlib import Path
import csv


def generate_csv(folder_path, train_data, test_data, train_img_paths_list, batch_idxs):

    if folder_path == "Leukemia":

        header_train = ["path", "label", "client_id"]

        with open("AML_train.csv", "w", encoding="UTF8") as fd:
            writer = csv.writer(fd)

            # write the header
            writer.writerow(header_train)

            # write multiple rows
            writer.writerows(train_data)

        header_test = ["path", "label"]

        with open("AML_test.csv", "w", encoding="UTF8") as f:
            writer = csv.writer(f)

            # write the header
            writer.writerow(header_test)

            # write multiple rows
            writer.writerows(test_data)

        header_dirichlet = ["path", "label", "batch_idx"]
        with open("AML_train_dirichlet.csv", "w+", encoding="UTF8") as fd:
            writer = csv.writer(fd)

            # write the header
            writer.writerow(header_train)

            for batch_idx, batch in enumerate(batch_idxs):
                for file_idx in batch:
                    file_path = train_img_paths_list[file_idx]
                    label = file_path[-16:-13]

                    writer.writerow([file_path, label, batch_idx])
