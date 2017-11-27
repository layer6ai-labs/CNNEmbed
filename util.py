import numpy as np

def pad_zeros(data_indices, zero_ind, max_doc_len):
    '''Pad the indices with zero in the beginning if the length is less than max number of words.'''

    new_data_indices = []
    for doc in data_indices:
        if len(doc) < max_doc_len:
            doc = [zero_ind for _ in range(max_doc_len - len(doc))] + doc
        new_data_indices.append(doc)

    return np.array(new_data_indices)