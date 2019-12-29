#!/usr/bin/env python
"""
Usage:
    test_step4.py [options] DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import build_vocab_from_data_dir, load_data_from_dir


def find_first(item, vector):
    """return the index of the first occurence of item in vector"""
    for i in range(len(vector)):
        if item == vector[i]:
            return i
    return len(vector)


def run(arguments) -> None:
    vocab = build_vocab_from_data_dir(
        arguments["DATA_DIR"],
        vocab_size=500,
        max_num_files=arguments.get("--max-num-files"),
    )
    tensorised_data = load_data_from_dir(
        vocab,
        length=50,
        data_dir=arguments["DATA_DIR"],
        max_num_files=arguments.get("--max-num-files"),
    )

    for idx in range(min(5, len(tensorised_data))):
        token_ids = tensorised_data[idx]
        length = find_first(
            vocab.get_id_or_unk(vocab.get_pad()), token_ids
        )
        tokens = [vocab.get_name_for_id(tok_id) for tok_id in token_ids]
        print("Sample %i:" % (idx))
        print(" Real length: %i" % (length))
        print(" Tensor length: %i" % (len(token_ids)))
        print(" Raw tensor: %s (truncated)" % (str(token_ids[: length + 2])))
        print(" Interpreted tensor: %s (truncated)" % (str(tokens[: length + 2])))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
