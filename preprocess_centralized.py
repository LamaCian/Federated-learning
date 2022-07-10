BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess_centralized(dataset):

    return (
        dataset.shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER)
    )
