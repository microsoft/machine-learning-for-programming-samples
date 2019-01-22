#!/usr/bin/env python
"""
Usage:
    test_step1.py [options] DATA_FILE

Options:
    -h --help                        Show this screen.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from model import Model


def run(arguments) -> None:
    model = Model({"run_id": "test step1"})
    print("Loaded token sequences:")
    for token_seq in model.load_data_file(arguments['DATA_FILE']):
        print(token_seq)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
