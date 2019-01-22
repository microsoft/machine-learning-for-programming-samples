#!/usr/bin/env python
"""
Usage:
    test_step2.py [options] DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
from docopt import docopt
from dpu_utils.utils import run_and_debug

from model import Model
from train import make_run_id

def run(arguments) -> None:
    hyperparameters = Model.get_default_hyperparameters()
    hyperparameters['run_id'] = make_run_id(arguments)
    model = Model(hyperparameters, model_save_dir=".")

    model.load_metadata_from_dir(arguments['DATA_DIR'],
                                 max_num_files=arguments.get('--max-num-files'))
    print("Loaded metadata for model: ")
    for key, value in model.metadata.items():
        print("  % 20s: %s" % (key, str(value)[:60]))

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
