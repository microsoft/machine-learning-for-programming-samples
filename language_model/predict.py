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
from typing import List

from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import tensorise_token_sequence, END_SYMBOL
from model import LanguageModel


def run(arguments) -> None:
    model = LanguageModel.restore(arguments["TRAINED_MODEL"])

    def compute_next_token(token_seq: List[str], num_cands: int = 3) -> str:
        tensorised_seq = tensorise_token_sequence(model.vocab, len(token_seq) + 1, token_seq)
        next_tok_probs = model.predict_next_token(tensorised_seq)
        top_idxs = (-next_tok_probs).argsort()[:num_cands]
        return [(model.vocab.get_name_for_id(top_idx),
                 next_tok_probs[top_idx])
                for top_idx in top_idxs]

    tokens = arguments['TOKENS']
    for idx in range(int(arguments['--num-steps'])):
        cands = compute_next_token(tokens)
        print("Prediction at step %i (tokens %s):" % (idx, tokens))
        for (token, prob) in cands:
            print(" Prob %.3f: %s" % (prob, token))
        next_tok = cands[0][0]
        if next_tok == END_SYMBOL:
            print('Reached end of sequence. Stopping.')
            break
        print("Continuing with token %s" % next_tok)
        tokens.append(next_tok)


if __name__ == '__main__':
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args['--debug'])
