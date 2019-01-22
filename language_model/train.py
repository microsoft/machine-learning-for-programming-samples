#!/usr/bin/env python
"""
Usage:
    train.py [options] SAVE_DIR TRAIN_DATA_DIR VALID_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help                        Show this screen.
    --max-num-epochs EPOCHS          The maximum number of epochs to run [default: 300]
    --max-num-files INT              Number of files to load.
    --hypers-override HYPERS         JSON dictionary overriding hyperparameter values.
    --run-name NAME                  Picks a name for the trained model.
    --debug                          Enable debug routines. [default: False]
"""
import json
import os
import sys
import time
from typing import Dict, Any, Optional

from docopt import docopt
from dpu_utils.utils import run_and_debug

from model import Model


def run_train(train_data_dir: str,
              valid_data_dir: str,
              save_dir: str,
              hyperparameters: Dict[str, Any],
              max_num_files: Optional[int]=None,
              parallelize: bool=True) \
        -> None:
    model = Model(hyperparameters, model_save_dir=save_dir)

    model.load_metadata_from_dir(train_data_dir, max_num_files=max_num_files)
    print("Loaded metadata for model: ")
    for key, value in model.metadata.items():
        print("  % 20s: %s" % (key, str(value)[:60]))

    train_data = model.load_data_from_dir(train_data_dir, max_num_files=max_num_files)
    print('Training on %i samples.' % len(train_data['tokens']))
    valid_data = model.load_data_from_dir(valid_data_dir, max_num_files=max_num_files)    
    print('Validating on %i samples.' % len(valid_data['tokens']))

    model.init()
    model.train(train_data, valid_data)


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    user_save_name = arguments.get('--run-name')
    if user_save_name is not None:
        user_save_name = user_save_name[:-len('.pkl')] if user_save_name.endswith('.pkl') else user_save_name
        return "%s" % (user_save_name)
    else:
        return "RNNModel-%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


def run(arguments) -> None:
    hyperparameters = Model.get_default_hyperparameters()
    hyperparameters['run_id'] = make_run_id(arguments)
    hyperparameters['max_epochs'] = int(arguments.get('--max-num-epochs'))

    # override hyperparams if flag is passed
    hypers_override = arguments.get('--hypers-override')
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))

    save_model_dir = args['SAVE_DIR']
    os.makedirs(save_model_dir, exist_ok=True)

    run_train(arguments['TRAIN_DATA_DIR'],
              arguments['VALID_DATA_DIR'],
              save_model_dir,
              hyperparameters,
              max_num_files=arguments.get('--max-num-files'))


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
