import numpy as np


def dirichlet(n_parties, total_count, beta):

    """Applies dirichlet distribution on train dataset

    Args:
        n_parties (int): The number of paritions that should be made
        total_count (int): total number of elements that should be distributed
        beta (int): parameter that controlls unabalance

    Returns:
        (list): indexes with a Dirichlet distribution
    """

    np.random.seed(8)
    idxs = np.random.permutation(total_count)
    min_size = 0

    while min_size < 1000:

        proportions = np.random.dirichlet(np.repeat(beta, n_parties))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * len(idxs))

    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]

    batch_idxs = np.split(idxs, proportions)

    return batch_idxs
