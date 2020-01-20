#!/usr/bin/env python
"""
Usage:
    train.py [options] SAVE_DIR TRAIN_DATA_DIR VALID_DATA_DIR

*_DATA_DIR are directories filled with files that we use as data.

Options:
    -h --help                        Show this screen.
    --max-num-epochs EPOCHS          The maximum number of epochs to run [default: 500]
    --patience NUM                   Number of epochs to wait for model improvement before stopping [default: 5]
    --max-num-files INT              Number of files to load.
    --hypers-override HYPERS         JSON dictionary overriding hyperparameter values.
    --run-name NAME                  Picks a name for the trained model.
    --debug                          Enable debug routines. [default: False]
"""
import json
import os
import time
from typing import Dict, Any

import numpy as np
from docopt import docopt
from dpu_utils.utils import run_and_debug

from dataset import build_vocab_from_data_dir, load_data_from_dir, get_minibatch_iterator
from model import LanguageModel

def train(
    model: LanguageModel,
    train_data: np.ndarray,
    valid_data: np.ndarray,
    batch_size: int,
    max_epochs: int,
    patience: int,
    save_file: str,
):
    best_valid_loss, _ = model.run_one_epoch(
        get_minibatch_iterator(valid_data, batch_size, is_training=False),
        training=False,
    )
    print(f"Initial valid loss: {best_valid_loss:.3f}.")
    model.save(save_file)
    best_valid_epoch = 0
    train_time_start = time.time()
    for epoch in range(1, max_epochs + 1):
        print(f"== Epoch {epoch}")
        train_loss, train_acc = model.run_one_epoch(
            get_minibatch_iterator(train_data, batch_size, is_training=True),
            training=True,
        )
        print(f" Train:  Loss {train_loss:.4f}, Acc {train_acc:.3f}")
        valid_loss, valid_acc = model.run_one_epoch(
            get_minibatch_iterator(valid_data, batch_size, is_training=False),
            training=False,
        )
        print(f" Valid:  Loss {valid_loss:.4f}, Acc {valid_acc:.3f}")

        # Save if good enough.
        if valid_loss < best_valid_loss:
            print(
                f"  (Best epoch so far, loss decreased {valid_loss:.4f} from {best_valid_loss:.4f})",
            )
            model.save(save_file)
            print(f"  (Saved model to {save_file})")
            best_valid_loss = valid_loss
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            print(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation loss.",
            )
            print(
                f"Training took {total_time:.0f}s. Best validation loss: {best_valid_loss:.4f}",
            )
            break


def run(arguments) -> None:
    hyperparameters = LanguageModel.get_default_hyperparameters()
    hyperparameters["run_id"] = make_run_id(arguments)
    max_epochs = int(arguments.get("--max-num-epochs"))
    patience = int(arguments.get("--patience"))
    max_num_files = arguments.get("--max-num-files")

    # override hyperparams if flag is passed
    hypers_override = arguments.get("--hypers-override")
    if hypers_override is not None:
        hyperparameters.update(json.loads(hypers_override))

    save_model_dir = args["SAVE_DIR"]
    os.makedirs(save_model_dir, exist_ok=True)
    save_file = os.path.join(
        save_model_dir, f"{hyperparameters['run_id']}_best_model.bin"
    )

    print("Loading data ...")
    vocab = build_vocab_from_data_dir(
        data_dir=args["TRAIN_DATA_DIR"],
        vocab_size=hyperparameters["max_vocab_size"],
        max_num_files=max_num_files,
    )
    print(f"  Built vocabulary of {len(vocab)} entries.")
    train_data = load_data_from_dir(
        vocab,
        length=hyperparameters["max_seq_length"],
        data_dir=args["TRAIN_DATA_DIR"],
        max_num_files=max_num_files,
    )
    print(f"  Loaded {train_data.shape[0]} training samples from {args['TRAIN_DATA_DIR']}.")
    valid_data = load_data_from_dir(
        vocab,
        length=hyperparameters["max_seq_length"],
        data_dir=args["VALID_DATA_DIR"],
        max_num_files=max_num_files,
    )
    print(f"  Loaded {valid_data.shape[0]} validation samples from {args['VALID_DATA_DIR']}.")
    model = LanguageModel(hyperparameters, vocab)
    model.build(([None, hyperparameters["max_seq_length"]]))
    print(
        f"Constructed model, using the following hyperparameters: {json.dumps(hyperparameters)}"
    )

    train(
        model,
        train_data,
        valid_data,
        batch_size=hyperparameters["batch_size"],
        max_epochs=max_epochs,
        patience=patience,
        save_file=save_file,
    )


def make_run_id(arguments: Dict[str, Any]) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    user_save_name = arguments.get("--run-name")
    if user_save_name is not None:
        user_save_name = (
            user_save_name[: -len(".pkl")]
            if user_save_name.endswith(".pkl")
            else user_save_name
        )
        return "%s" % (user_save_name)
    else:
        return "RNNModel-%s" % (time.strftime("%Y-%m-%d-%H-%M-%S"))


if __name__ == "__main__":
    args = docopt(__doc__)
    run_and_debug(lambda: run(args), args["--debug"])
