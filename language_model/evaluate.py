#!/usr/bin/env python
"""
Usage:
    evaluate.py [options] TRAINED_MODEL TEST_DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import load_data_from_dir, get_minibatch_iterator
from model import LanguageModel


def run(arguments) -> None:
    print("Loading data ...")
    model = LanguageModel.restore(arguments["TRAINED_MODEL"])
    print(f"  Loaded trained model from {arguments['TRAINED_MODEL']}.")

    test_data = load_data_from_dir(
        model.vocab,
        length=model.hyperparameters["max_seq_length"],
        data_dir=arguments["TEST_DATA_DIR"],
        max_num_files=arguments.get("--max-num-files"),
    )
    print(
        f"  Loaded {test_data.shape[0]} test samples from {arguments['TEST_DATA_DIR']}."
    )

    test_loss, test_acc = model.run_one_epoch(
        get_minibatch_iterator(
            test_data,
            model.hyperparameters["batch_size"],
            is_training=False,
            drop_remainder=False,
        ),
        training=False,
    )
    print(f"Test:  Loss {test_loss:.4f}, Acc {test_acc:.3f}")


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
