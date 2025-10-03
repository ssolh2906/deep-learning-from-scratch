import numpy as np

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
    Convert sequence of length L (60) to (L=60,4) array X
    """
    try:
        X = np.stack([encode_base(ch) for ch in seq], axis=0).astype(np.float32)
        return X
    except Exception as e:
        raise Exception(f"Failed to encode sequence: {e}")