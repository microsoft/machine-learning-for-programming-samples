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

from model import Model
from train import make_run_id

def run(arguments) -> None:
    hyperparameters = Model.get_default_hyperparameters()
    hyperparameters['run_id'] = make_run_id(arguments)
    model = Model(hyperparameters, model_save_dir=".")

    max_num_files = arguments.get('--max-num-files')
    data_dir = arguments['DATA_DIR']
    model.load_metadata_from_dir(data_dir, max_num_files=max_num_files)
    data = model.load_data_from_dir(data_dir, max_num_files=max_num_files)

    for idx in range(min(5, len(data['tokens']))):
        length = data['tokens_lengths'][idx]
        token_ids = data['tokens'][idx]
        tokens = [model.metadata['token_vocab'].get_name_for_id(tok_id) for tok_id in token_ids]
        print("Sample %i:" % (idx))
        print(" Real length: %i" % (length))
        print(" Tensor length: %i" % (len(token_ids)))
        print(" Raw tensor: %s (truncated)" % (str(token_ids[:length+2])))
        print(" Interpreted tensor: %s (truncated)" % (str(tokens[:length+2])))

if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
