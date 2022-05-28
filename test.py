import pathlib
import tensorflow as tf
from pathlib import Path
import numpy as np


dataset = pathlib.Path("Leukemia")
data = "Leukemia"

list2 = tf.data.Dataset.list_files(data)
for i in list2.take(2):
    print(i.numpy())

list_ds = tf.data.Dataset.list_files(str(dataset / "*/*"))

for f in list_ds.take(5):
    print(f.numpy())
