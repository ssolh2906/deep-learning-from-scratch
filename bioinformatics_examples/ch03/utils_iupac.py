import numpy as np
'''
utils_iupac

Encode '60- mer IUPAC characters' to 'one-hot/frac-hot(60x4)'
Flat (60x4) to (240,)
'''

# I wrote this code down with the help of ChatGPT5.0


BASES = ['A', 'C', 'G', 'T']
BASES_TO_INDEX = {b:i for i, b in enumerate(BASES)}

# IUPAC table
IUPAC = {
    'A': ['A'], 'C': ['C'], 'G': ['G'], 'T': ['T'],
    'R': ['A', 'G'],
    'Y': ['C', 'T'],
    'S': ['G', 'C'],
    'W': ['A', 'T'],
    'K': ['G', 'T'],
    'M': ['A', 'C'],
    'B': ['C', 'G', 'T'],
    'D': ['A', 'G', 'T'],
    'H': ['A', 'C', 'T'],
    'V': ['A', 'C', 'G'],
    'N': ['A', 'C', 'G', 'T']
}


def encode_base(base: str) -> np.ndarray:
    """
    Base -> vector
    ex) A = [1,0,0,0]
        T = [0,0,0,1]
        K = [0,0,0.5,0.5]
    """
    base = base.upper()
    vector = np.zeros(4, dtype=np.float32) # np.float32 for Deep learning compatibility

    # If base is A,C,G,T
    # One-hot vector
    if base in BASES_TO_INDEX:
        vector[BASES_TO_INDEX[base]] = 1.0
        return vector

    # If base is ambiguous, frac-hot vector.
    # Give uniform weights to ambiguous bases.
    if base in IUPAC:
        choices = IUPAC[base]
        w = 1.0 / len(choices)

        for c in choices:
            vector[BASES_TO_INDEX[c]] = w
        return vector

    # Unexpected input _> [0,0,0,0]
    return vector


def encode_seq(seq: str) -> np.ndarray:
    """
    Convert sequence of length L (60) to (L=60,4) array seq_matrix
    """
    try:
        seq_matrix = np.stack([encode_base(ch) for ch in seq], axis=0).astype(np.float32)
        return seq_matrix
    except Exception as e:
        raise Exception("Failed to encode sequence: {e}")


def flatten_encoded_seq(seq_matrix: np.ndarray) -> np.ndarray:
    try:
        return seq_matrix.reshape(-1).astype(np.float32) # -1: Infer this dimension, in this case, number of all elements.
    except Exception as e:
        raise Exception(f"Failed to flatten sequence: {e}")


# test
if __name__ == '__main__':
    seq = "ATTYCGGRTG"*10 # Length 60
    print("Sequence:",seq[:10],"...")
    print()

    seq_matrix = encode_seq(seq)
    print("Encoded seq shape:", seq_matrix.shape)
    print(seq_matrix[:5])
    print()

    x = flatten_encoded_seq(seq_matrix)
    print("Flattened seq shape:", x.shape)

