import os.path
import pickle

import numpy as np
from ucimlrepo import fetch_ucirepo

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/splicing_pwm.pkl"


def _load_raw_from_ucirepo():  # init_mnist
    """
    Fetch UCI Splice-junction Gene Sequence dataset using ucimlrepo package. (id = 69).
    Return list_x., list_t
    """

    # fetch dataset
    splice_junction_seqs = fetch_ucirepo(id=69)

    # data (as pandas dataframes)
    list_x = splice_junction_seqs.data.features
    list_t = splice_junction_seqs.data.targets

    # metadata
    print(splice_junction_seqs.metadata)

    # variable information
    print(splice_junction_seqs.variables)

    return list_x, list_t


_SYMBOL_TO_PROBABILITY = {
    'A': [1.0, 0.0, 0.0, 0.0],
    'G': [0.0, 1.0, 0.0, 0.0],
    'T': [0.0, 0.0, 1.0, 0.0],
    'C': [0.0, 0.0, 0.0, 1.0],
    'D': [1 / 3, 1 / 3, 1 / 3, 0],  # AGT
    'N': [0.25, 0.25, 0.25, 0.25],  # AGCT
    'S': [0.0, 0.5, 0.0, 0.5],  # GC
    'R': [0.5, 0.5, 0.0, 0.0],  # AG
}


def seq_to_prob_flatten(list_x):
    arr = np.asarray(list_x).astype(str)
    n, L = arr.shape
    out = np.zeros((n, L, 4), dtype=float)
    for i in range(L):
        col = arr[:, i]
        for j, ch in enumerate(col):
            prob = _SYMBOL_TO_PROBABILITY.get(ch, None)
            if prob is not None:
                out[j, i, :] = prob
            else:
                out[j, i, :] = 0
    return out.reshape(n, -1)


def _convert_numpy():
    """
    Fetch the raw data, covert to flattened PWM inputs and labels.
    Split train/test sets.
    Return dataset dictionary.
    """
    list_x, list_t = _load_raw_from_ucirepo()
    X = seq_to_prob_flatten(list_x)
    y = np.asarray(list_t).ravel()

    rng = np.random.RandomState(29)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    split = int(len(X) * 0.8)
    train_idx = idx[:split]
    test_idx = idx[split:]

    dataset = {
        'train_pwm': X[train_idx],
        'train_label': y[train_idx],
        'test_pwm': X[test_idx],
        'test_label': y[test_idx]
    }

    return dataset


def init_pwm():
    """
    Create and save the processed pwm dataset as a pickle file.
    """
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)
    print("Pickle file saved")


def load_pwm():
    """
    Load Splice-junction Gene Sequence dataset from pickle file.
    data: one-hot / frac encoding, flattened ndarray
    """

    if not os.path.exists(save_file):
        init_pwm()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    return (dataset['train_pwm'], dataset['train_label']), (dataset['test_pwm'], dataset['test_label'])


if __name__ == '__main__':
    load_pwm()
