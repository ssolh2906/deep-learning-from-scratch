"""
A loader for the UCI Splice-junction Gene Sequence dataset.
"""
import numpy as np
import logging
from typing import List, Tuple
from bioinformatics_examples.ch03.utils_iupac import flatten_encoded_seq
from utils_iupac import encode_seq

LABEL_MAP = {
    'EI': 0,
    'IE': 1,
    'N':2
}

logger = logging.getLogger(__name__)

def _parse_line(line) -> tuple[int,str]:
    """
    Parse a single pair of label and sequence from a line.
    """
    line = (line or "").strip()

    raw_parts = line.split(',')
    parts = []
    for part in raw_parts:
        token = part.strip()
        if token:
            parts.append(token)

    label = parts[0].upper()
    seq = parts[-1].upper()

    return LABEL_MAP[label], seq

def seq_to_vector(seq):
    """
    Convert 60-mer IUPAC sequence to vector (240,).
    """
    matrix = encode_seq(seq)
    vector = flatten_encoded_seq(matrix)
    return vector

def _load_raw(path):
    """
    Read the dataset file and return list of vectors and labels.
    vector: list of (240,)
    labels: list of labels {EI:0, IE:1, N:2}
    """
    n_skipped = 0
    list_x:List[np.ndarray] = []
    list_t:List[int] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                t , x = _parse_line(line)
                vec = seq_to_vector(x)
                list_x.append(vec)
                list_t.append(t)
            except Exception:
                n_skipped += 1
                continue
        if n_skipped > 0:
            logging.warning(f"Skipped {n_skipped} lines")
    return (list_x, list_t)

def _split_indicies(n, seed = 0):
    """
    Return (train_idx, val_idx, test_idx) lists.
    """
    split = (0.6, 0.6, 0.6)
    a,b,c = split

    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(n * a)
    n_val = int(n * b)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train+n_val]
    test_idx = idx[n_train+n_val:]

    return (train_idx, val_idx, test_idx)


def load_splice(path, seed = 0):
    xs, ts = _load_raw(path)
    matrix_x = np.vstack(xs).astype(np.float32)
    matrix_t = np.asarray(ts, dtype=np.int64)

    train_idx, val_idx, test_idx = _split_indicies(len(matrix_x),seed)
    def _take(indices):
        return matrix_x[indices], matrix_t[indices]

    return _take(train_idx), _take(val_idx), _take(test_idx)

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    # Test _parse_line
    line_ok = "IE, sample_001, ACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTN"
    t,seq = _parse_line(line_ok)
    print("label ID:" ,t)
    print("Sequence:", seq)

    # Test seq_to_vector
    seq = "ACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTNACGTN"
    vector = seq_to_vector(seq)
    print("Vector shape:", vector.shape)
    print("Vector first 16 values:", vector[:16])

    # TODO
    """
    From part of real data, get short test data
    """
    # Test load splice
    # test_data_path = "bioinformatics_examples\ch03\test_data\splice_test.data"