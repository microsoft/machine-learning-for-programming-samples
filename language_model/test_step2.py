#!/usr/bin/env python
"""
Usage:
    test_step2.py [options] DATA_FILE

Options:
    -h --help                        Show this screen.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import load_data_file

def run(arguments) -> None:
    print("Loaded token sequences:")
    for token_seq in load_data_file(arguments['DATA_FILE']):
        print(token_seq)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
