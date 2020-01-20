import os
from glob import iglob
from typing import List, Dict, Any, Iterable, Optional, Iterator

import numpy as np
from more_itertools import chunked
from dpu_utils.mlutils.vocabulary import Vocabulary


DATA_FILE_EXTENSION = "proto"
START_SYMBOL = "%START%"
END_SYMBOL = "%END%"


def get_data_files_from_directory(
    data_dir: str, max_num_files: Optional[int] = None
) -> List[str]:
    files = iglob(
        os.path.join(data_dir, "**/*.%s" % DATA_FILE_EXTENSION), recursive=True
    )
    if max_num_files:
        files = sorted(files)[: int(max_num_files)]
    else:
        files = list(files)
    return files


def load_data_file(file_path: str) -> Iterable[List[str]]:
    """
    Load a single data file, returning token streams.

    Args:
        file_path: The path to a data file.

    Returns:
        Iterable of lists of strings, each a list of tokens observed in the data.
    """
    #TODO 2# Insert your data parsing code here
    return TODO


def build_vocab_from_data_dir(
    data_dir: str, vocab_size: int, max_num_files: Optional[int] = None
) -> Vocabulary:
    """
    Compute model metadata such as a vocabulary.

    Args:
        data_dir: Directory containing data files.
        vocab_size: Maximal size of the vocabulary to create.
        max_num_files: Maximal number of files to load.
    """

    data_files = get_data_files_from_directory(data_dir, max_num_files)

    vocab = Vocabulary(add_unk=True, add_pad=True)
    # Make sure to include the START_SYMBOL in the vocabulary as well:
    vocab.add_or_get_id(START_SYMBOL)
    vocab.add_or_get_id(END_SYMBOL)

    #TODO 3# Insert your vocabulary-building code here

    return vocab


def tensorise_token_sequence(
    vocab: Vocabulary, length: int, token_seq: Iterable[str],
) -> List[int]:
    """
    Tensorise a single example.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        token_seq: Sequence of tokens to tensorise.

    Returns:
        List with length elements that are integer IDs of tokens in our vocab.
    """
    #TODO 4# Insert your tensorisation code here
    return TODO


def load_data_from_dir(
    vocab: Vocabulary, length: int, data_dir: str, max_num_files: Optional[int] = None
) -> np.ndarray:
    """
    Load and tensorise data.

    Args:
        vocab: Vocabulary to use for mapping tokens to integer IDs
        length: Length to truncate/pad sequences to.
        data_dir: Directory from which to load the data.
        max_num_files: Number of files to load at most.

    Returns:
        numpy int32 array of shape [None, length], containing the tensorised
        data.
    """
    data_files = get_data_files_from_directory(data_dir, max_num_files)
    data = np.array(
        list(
            tensorise_token_sequence(vocab, length, token_seq)
            for data_file in data_files
            for token_seq in load_data_file(data_file)
        ),
        dtype=np.int32,
    )
    return data


def get_minibatch_iterator(
    token_seqs: np.ndarray,
    batch_size: int,
    is_training: bool,
    drop_remainder: bool = True,
) -> Iterator[np.ndarray]:
    indices = np.arange(token_seqs.shape[0])
    if is_training:
        np.random.shuffle(indices)

    for minibatch_indices in chunked(indices, batch_size):
        if len(minibatch_indices) < batch_size and drop_remainder:
            break  # Drop last, smaller batch

        minibatch_seqs = token_seqs[minibatch_indices]
        yield minibatch_seqs
