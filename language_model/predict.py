#!/usr/bin/env python
"""
Usage:
    predict.py [options] TRAINED_MODEL TOKENS...

Uses trained model to continue the sequence of tokens provided.

Options:
    -h --help        Show this screen.
    --num-steps NUM  Number of steps to continue token sequence for. [default: 5]
    --debug          Enable debug routines. [default: False]
"""
import json
import os
import sys
import time
from typing import Dict, Any, Optional, List

import numpy as np
from docopt import docopt
from dpu_utils.utils import git_tag_run, run_and_debug

from evaluate import restore


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def run(arguments) -> None:
    model = restore(arguments['TRAINED_MODEL'])

    def compute_next_token(token_seq: List[str], num_cands: int = 3) -> str:
        loaded_sample = model.load_data_from_raw_sample_sequences([token_seq])
        feed_dict = {
            model.placeholders['dropout_keep_rate']: 1.0,
            model.placeholders['tokens']: loaded_sample['tokens'],
            model.placeholders['tokens_lengths']: loaded_sample['tokens_lengths'],
        }
        output_logits = model.sess.run(model.ops['output_logits'], feed_dict=feed_dict)
        next_tok_logits = output_logits[0,loaded_sample['tokens_lengths'] - 1][0]
        next_tok_probs = softmax(next_tok_logits)
        top_idxs = (-next_tok_probs).argsort()[:num_cands]
        return [(model.metadata['token_vocab'].get_name_for_id(top_idx),
                 next_tok_probs[top_idx])
                for top_idx in top_idxs]

    tokens = arguments['TOKENS']
    for idx in range(int(arguments['--num-steps'])):
        cands = compute_next_token(tokens)
        print("Prediction at step %i (tokens %s):" % (idx, tokens))
        for (token, prob) in cands:
            print(" Prob %.3f: %s" % (prob, token))
        next_tok = cands[0][0]
        print("Continuing with token %s" % next_tok)
        tokens.append(next_tok)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
