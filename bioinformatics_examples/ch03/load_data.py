from ucimlrepo import fetch_ucirepo


def _load_raw_from_ucirepo():
    '''
    Fetch UCI Splice-junction Gene Sequence dataset using ucimlrepo package. (id = 69).
    Return list_x., list_t
    '''

    # fetch dataset
    molecular_biology_splice_junction_gene_sequences = fetch_ucirepo(id=69)

    # data (as pandas dataframes)
    list_x = molecular_biology_splice_junction_gene_sequences.data.features
    list_t = molecular_biology_splice_junction_gene_sequences.data.targets

    # metadata
    print(molecular_biology_splice_junction_gene_sequences.metadata)

    # variable information
    print(molecular_biology_splice_junction_gene_sequences.variables)

    return (list_x, list_t)
