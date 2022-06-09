NUM_EPOCHS = 5
BATCH_SIZE = 20  # faster training bigger batch size, >100 if 32x32,
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10


def preprocess(dataset):

    return (
        dataset.repeat(NUM_EPOCHS)
        .shuffle(SHUFFLE_BUFFER, seed=1)
        .batch(BATCH_SIZE)
        .prefetch(PREFETCH_BUFFER)
    )
