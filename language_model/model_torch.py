import gzip
import os
from typing import Dict, Any, NamedTuple, Iterable, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dpu_utils.mlutils.vocabulary import Vocabulary


class LanguageModelLoss(NamedTuple):
    token_ce_loss: torch.Tensor
    num_predictions: torch.Tensor
    num_correct_token_predictions: torch.Tensor


class LanguageModelTorch(nn.Module):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
            "learning_rate": 0.01,
            "learning_rate_decay": 0.98,
            "momentum": 0.85,
            "max_epochs": 500,
            "patience": 5,
            "max_vocab_size": 10000,
            "max_seq_length": 50,
            "batch_size": 200,
            "token_embedding_size": 64,
            "rnn_hidden_dim": 64,
            "use_gpu": True,
        }

    def __init__(self, hyperparameters: Dict[str, Any], vocab: Vocabulary,) -> None:
        self.hyperparameters = hyperparameters
        self.vocab = vocab
        self.optimizer = None  # Will be built later

        if torch.cuda.is_available() and self.hyperparameters["use_gpu"]:
            print("Running model on GPU.")
            self.device = torch.device("cuda:0")
        else:
            print("Running model on CPU.")
            self.device = torch.device("cpu")

        super().__init__()

    @property
    def run_id(self):
        return self.hyperparameters["run_id"]

    def save(self, path: str) -> None:
        with gzip.open(path, "wb") as out_file:
            torch.save(self, out_file)

    @classmethod
    def restore(cls, saved_model_path: str) -> "LanguageModelTorch":
        with gzip.open(saved_model_path, "rb") as fh:
            return torch.load(fh)

    def build(self, input_shape):
        emb_dim = self.hyperparameters["token_embedding_size"]
        rnn_dim = self.hyperparameters["rnn_hidden_dim"]

        # TODO 5# Build necessary submodules here

        if torch.cuda.is_available() and self.hyperparameters["use_gpu"]:
            self.cuda()
        else:
            self.cpu()

    def forward(self, inputs):
        return self.compute_logits(inputs)

    def compute_logits(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Implements a language model, where each output is conditional on the current
        input and inputs processed so far.

        Args:
            inputs: int32 tensor of shape [B, T], storing integer IDs of tokens.

        Returns:
            torch.float32 tensor of shape [B, T, V], storing the distribution over output symbols
            for each timestep for each batch element.
        """
        # TODO 5# 1) Embed tokens
        # TODO 5# 2) Run RNN on embedded tokens
        # TODO 5# 3) Project RNN outputs onto the vocabulary to obtain logits.
        return rnn_output_logits

    def compute_loss_and_acc(
        self, rnn_output_logits: torch.Tensor, target_token_seq: torch.Tensor
    ) -> LanguageModelLoss:
        """
        Args:
            rnn_output_logits: torch.float32 Tensor of shape [B, T, V], representing
                logits as computed by the language model.
            target_token_seq: torch.int32 Tensor of shape [B, T], representing
                the target token sequence.

        Returns:
            LanguageModelLoss tuple, containing both the average per-token loss
            as well as the number of (non-padding) token predictions and how many
            of those were correct.
        
        Note:
            We assume that the two inputs are shifted by one from each other, i.e.,
            that rnn_output_logits[i, t, :] are the logits for sample i after consuming
            input t; hence its target output is assumed to be target_token_seq[i, t+1].
        """
        # TODO 5# 4) Compute CE loss for all but the last timestep:
        token_ce_loss = TODO

        # TODO 6# Compute number of (correct) predictions
        num_tokens = torch.zeros([])
        num_correct_tokens = torch.zeros([])

        # TODO 7# Mask out CE loss for padding tokens

        return LanguageModelLoss(token_ce_loss, num_tokens, num_correct_tokens)

    def predict_next_token(self, token_seq: List[int]):
        self.eval()
        inputs = torch.tensor([token_seq], dtype=torch.long, device=self.device)
        output_logits = self.compute_logits(inputs)
        next_tok_logits = output_logits[0, -1, :]
        next_tok_probs = torch.nn.functional.softmax(next_tok_logits, dim=0)
        return next_tok_probs.detach().cpu().numpy()

    def _make_optimizer(self):
        if self.optimizer is not None:
            return

        # Also prepare optimizer:
        optimizer_name = self.hyperparameters["optimizer"].lower()
        if optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                params=self.parameters(),
                lr=self.hyperparameters["learning_rate"],
                momentum=self.hyperparameters["momentum"],
            )
        elif optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(
                params=self.parameters(),
                lr=self.hyperparameters["learning_rate"],
                alpha=self.params["learning_rate_decay"],
                momentum=self.params["momentum"],
            )
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(
                params=self.parameters(), lr=self.hyperparameters["learning_rate"],
            )
        else:
            raise Exception('Unknown optimizer "%s".' % (self.params["optimizer"]))

    def run_one_epoch(
        self, minibatches: Iterable[np.ndarray], training: bool = False,
    ):
        total_loss, num_samples, num_tokens, num_correct_tokens = 0.0, 0, 0, 0
        if training:
            self._make_optimizer()
            self.train()
        else:
            self.eval()

        for step, minibatch_data in enumerate(minibatches):
            if training:
                self.optimizer.zero_grad()
            minibatch_data = torch.tensor(
                minibatch_data, dtype=torch.long, device=self.device
            )
            model_outputs = self.compute_logits(minibatch_data)
            result = self.compute_loss_and_acc(model_outputs, minibatch_data)

            total_loss += result.token_ce_loss.item()
            num_samples += minibatch_data.shape[0]
            num_tokens += result.num_predictions.item()
            num_correct_tokens += result.num_correct_token_predictions.item()

            if training:
                result.token_ce_loss.backward()
                self.optimizer.step()

            print(
                "   Batch %4i: Epoch avg. loss: %.5f || Batch loss: %.5f | acc: %.5f"
                % (
                    step,
                    total_loss / num_samples,
                    result.token_ce_loss,
                    result.num_correct_token_predictions
                    / (float(result.num_predictions) + 1e-7),
                ),
                end="\r",
            )
        print("\r\x1b[K", end="")
        return (
            total_loss / num_samples,
            num_correct_tokens / float(num_tokens + 1e-7),
        )
