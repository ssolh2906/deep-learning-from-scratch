"""
A loader for the UCI Splice-junction Gene Sequence dataset.
"""
from logging import raiseExceptions

import numpy as np

from bioinformatics_examples.ch03.utils_iupac import flatten_encoded_seq
from utils_iupac import encode_seq

LABEL_MAP = {
    'EI': 0,
    'IE': 1,
    'N':2
}


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
    n_total = 0
    n_skipped = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                t , seq = _parse_line(line)
                x = encode_seq(seq)
                n_total += 1
            except Exception as e:
                raiseExceptions(e)
    return 0


if __name__ == '__main__':
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