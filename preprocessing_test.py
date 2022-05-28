import tensorflow as tf
import pandas as pd

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
labels_tf = tf.convert_to_tensor(labels)

# parts = tf.string.split("Leukemia/NGS/NGS_4868.jpg")


img = tf.io.read_file("Leukemia/NGS/NGS_4868.jpg")
print(img)


def parse_image(filename):

    """Transforms images into right shape for federated training

    Args:
        filename (_type_): _description_

    Returns:
        _type_: Image transformed into tensor
    """

    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image, channels=3)  #
    image = tf.image.convert_image_dtype(image, tf.float32)

    return image


def create_dataset(client_id):

    """create_dataset _summary_

    Args:
        client_id (_type_): _description_

    Returns:
        _type_: _description_
    """
    df = pd.read_csv("AML_train_dirichlet.csv", encoding="utf-8")

    client_id = int(client_id)

    file = df.loc[df["client_id"] == client_id]
    # print(file)
    path = file["path"]

    # print(path)
    list_ds = tf.data.Dataset.list_files(path)

    images_ds = list_ds.map(parse_image)

    return images_ds

 client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids, create_dataset
    )
