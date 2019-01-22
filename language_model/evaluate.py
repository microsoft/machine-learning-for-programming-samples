#!/usr/bin/env python
"""
Usage:
    evaluate.py [options] TRAINED_MODEL TEST_DATA_DIR

Options:
    -h --help                        Show this screen.
    --max-num-files INT              Number of files to load.
    --debug                          Enable debug routines. [default: False]
"""
import gzip
import json
import os
import pickle
import sys
import time
from typing import Dict, Any, Optional

import tensorflow as tf
from docopt import docopt
from dpu_utils.utils import run_and_debug

from model import Model


def restore(path: str) -> Model:
    with gzip.open(path) as f:
        saved_data = pickle.load(f)
    model = Model(saved_data['hyperparameters'], saved_data.get('run_name'))
    model.metadata.update(saved_data['metadata'])
    model.init()

    variables_to_initialize = []
    with model.sess.graph.as_default():
        with tf.name_scope("restore"):
            restore_ops = []
            used_vars = set()
            for variable in sorted(model.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES), key=lambda v: v.name):
                used_vars.add(variable.name)
                if variable.name in saved_data['weights']:
                    # print('Initializing %s from saved value.' % variable.name)
                    restore_ops.append(variable.assign(saved_data['weights'][variable.name]))
                else:
                    print('Freshly initializing %s since no saved value was found.' % variable.name)
                    variables_to_initialize.append(variable)
            for var_name in sorted(saved_data['weights']):
                if var_name not in used_vars:
                    if var_name.endswith('Adam:0') or var_name.endswith('Adam_1:0') or var_name in ['beta1_power:0', 'beta2_power:0']:
                        continue
                    print('Saved weights for %s not used by model.' % var_name)
            restore_ops.append(tf.variables_initializer(variables_to_initialize))
            model.sess.run(restore_ops)
    return model


def run(arguments) -> None:
    model = restore(arguments['TRAINED_MODEL'])
    test_data = model.load_data_from_dir(arguments['TEST_DATA_DIR'],
                                         max_num_files=arguments.get('--max-num-files'))
    model.test(test_data)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
