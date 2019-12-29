#!/usr/bin/env python
"""
Usage:
    test_step3.py [options] DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import build_vocab_from_data_dir


def run(arguments) -> None:
    vocab = build_vocab_from_data_dir(
        arguments["DATA_DIR"],
        vocab_size=500,
        max_num_files=arguments.get("--max-num-files")
    )

    print("Loaded vocabulary for dataset: ")
    print(" %s [...]" % (str(vocab)[:100]))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
