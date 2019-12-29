import os
import gzip
import pickle
from typing import Dict, Any, NamedTuple, Iterable, List

import numpy as np
import tensorflow.compat.v1 as tf
from dpu_utils.mlutils.vocabulary import Vocabulary


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
tf.get_logger().setLevel("ERROR")


class LanguageModelLoss(NamedTuple):
    token_ce_loss: tf.Tensor
    num_predictions: tf.Tensor
    num_correct_token_predictions: tf.Tensor


class LanguageModelTF1(object):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "optimizer": "Adam",  # One of "SGD", "RMSProp", "Adam"
            "learning_rate": 0.01,
            "learning_rate_decay": 0.98,
            "momentum": 0.85,
            "gradient_clip_value": 1,
            "max_epochs": 500,
            "patience": 5,
            "max_vocab_size": 10000,
            "max_seq_length": 50,
            "batch_size": 200,
            "token_embedding_size": 64,
            "rnn_hidden_dim": 64,
        }

    def __init__(self, hyperparameters: Dict[str, Any], vocab: Vocabulary,) -> None:
        self.hyperparameters = hyperparameters
        self.vocab = vocab
        self._sess = tf.Session(graph=tf.Graph())
        self._placeholders = {}
        self._weights = {}
        self._ops = {}

        super().__init__()

    @property
    def run_id(self):
        return self.hyperparameters["run_id"]

    def save(self, path: str) -> None:
        variables_to_save = list(
            set(self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        )
        weights_to_save = self._sess.run(variables_to_save)
        weights_to_save = {
            var.name: value for (var, value) in zip(variables_to_save, weights_to_save)
        }

        data_to_save = {
            "model_type": self.__class__.__name__,
            "hyperparameters": self.hyperparameters,
            "vocab": self.vocab,
            "weights": weights_to_save,
            "run_id": self.run_id,
        }

        with gzip.GzipFile(path, "wb") as outfile:
            pickle.dump(data_to_save, outfile)

    @classmethod
    def restore(cls, saved_model_path: str) -> "LanguageModelTF1":
        with gzip.open(saved_model_path) as f:
            saved_data = pickle.load(f)
        model = cls(saved_data["hyperparameters"], saved_data["vocab"])
        model.build((None, None))

        variables_to_initialize = []
        with model._sess.graph.as_default():
            with tf.name_scope("restore"):
                restore_ops = []
                used_vars = set()
                for variable in sorted(
                    model._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES),
                    key=lambda v: v.name,
                ):
                    used_vars.add(variable.name)
                    if variable.name in saved_data["weights"]:
                        # print('Initializing %s from saved value.' % variable.name)
                        restore_ops.append(
                            variable.assign(saved_data["weights"][variable.name])
                        )
                    else:
                        print(
                            "Freshly initializing %s since no saved value was found."
                            % variable.name
                        )
                        variables_to_initialize.append(variable)
                for var_name in sorted(saved_data["weights"]):
                    if var_name not in used_vars:
                        if (
                            var_name.endswith("Adam:0")
                            or var_name.endswith("Adam_1:0")
                            or var_name in ["beta1_power:0", "beta2_power:0"]
                        ):
                            continue
                        print("Saved weights for %s not used by model." % var_name)
                restore_ops.append(tf.variables_initializer(variables_to_initialize))
                model._sess.run(restore_ops)
        return model

    def build(self, input_shape):
        with self._sess.graph.as_default():
            self._placeholders["tokens"] = tf.placeholder(
                dtype=tf.int32, shape=[None, None], name="tokens"
            )

            self._ops["output_logits"] = self.compute_logits(
                self._placeholders["tokens"]
            )
            self._ops["output_probs"] = tf.nn.softmax(self._ops["output_logits"], -1)
            result = self.compute_loss_and_acc(
                rnn_output_logits=self._ops["output_logits"],
                target_token_seq=self._placeholders["tokens"],
            )
            self._ops["loss"] = result.token_ce_loss
            self._ops["num_tokens"] = result.num_predictions
            self._ops["num_correct_tokens"] = result.num_correct_token_predictions
            self._ops["train_step"] = self._make_training_step(self._ops["loss"])

            init_op = tf.variables_initializer(
                self._sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            )
            self._sess.run(init_op)

    def compute_logits(self, token_ids: tf.Tensor) -> tf.Tensor:
        """
        Implements a language model, where each output is conditional on the current
        input and inputs processed so far.

        Args:
            token_ids: int32 tensor of shape [B, T], storing integer IDs of tokens.

        Returns:
            tf.float32 tensor of shape [B, T, V], storing the distribution over output symbols
            for each timestep for each batch element.
        """
        # TODO 5# 1) Embed tokens
        # TODO 5# 2) Run RNN on embedded tokens
        # TODO 5# 3) Project RNN outputs onto the vocabulary to obtain logits.
        return rnn_output_logits

    def compute_loss_and_acc(
        self, rnn_output_logits: tf.Tensor, target_token_seq: tf.Tensor
    ) -> LanguageModelLoss:
        """
        Args:
            rnn_output_logits: tf.float32 Tensor of shape [B, T, V], representing
                logits as computed by the language model.
            target_token_seq: tf.int32 Tensor of shape [B, T], representing
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
        num_tokens = tf.constant(0)
        num_correct_tokens = tf.constant(0)

        # TODO 7# Mask out CE loss for padding tokens

        return LanguageModelLoss(token_ce_loss, num_tokens, num_correct_tokens)

    def predict_next_token(self, token_seq: List[int]):
        feed_dict = {
            self._placeholders["tokens"]: [token_seq],
        }
        output_probs = self._sess.run(self._ops["output_probs"], feed_dict=feed_dict)
        next_tok_probs = output_probs[0, -1, :]
        return next_tok_probs

    def _make_training_step(self, loss: tf.Tensor) -> tf.Tensor:
        """
        Constructs a trainig step from the loss parameter and hyperparameters.
        """
        optimizer_name = self.hyperparameters["optimizer"].lower()
        if optimizer_name == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.hyperparameters["learning_rate"]
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.hyperparameters["learning_rate"],
                decay=self.hyperparameters["learning_rate_decay"],
                momentum=self.hyperparameters["momentum"],
            )
        elif optimizer_name == "adam":
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.hyperparameters["learning_rate"]
            )
        else:
            raise Exception(
                'Unknown optimizer "%s".' % (self.hyperparameters["optimizer"])
            )

        # Calculate and clip gradients
        trainable_vars = self._sess.graph.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES
        )
        gradients = tf.gradients(loss, trainable_vars)
        clipped_gradients, _ = tf.clip_by_global_norm(
            gradients, self.hyperparameters["gradient_clip_value"]
        )
        pruned_clipped_gradients = []
        for (gradient, trainable_var) in zip(clipped_gradients, trainable_vars):
            if gradient is None:
                continue
            pruned_clipped_gradients.append((gradient, trainable_var))
        return optimizer.apply_gradients(pruned_clipped_gradients)

    def run_one_epoch(
        self, minibatches: Iterable[np.ndarray], training: bool = False,
    ):
        total_loss, num_samples, num_tokens, num_correct_tokens = 0.0, 0, 0, 0
        for step, minibatch_data in enumerate(minibatches):
            ops_to_run = {
                "loss": self._ops["loss"],
                "num_tokens": self._ops["num_tokens"],
                "num_correct_tokens": self._ops["num_correct_tokens"],
            }
            if training:
                ops_to_run["train_step"] = self._ops["train_step"]
            op_results = self._sess.run(
                ops_to_run, feed_dict={self._placeholders["tokens"]: minibatch_data}
            )
            total_loss += op_results["loss"]
            num_samples += minibatch_data.shape[0]
            num_tokens += op_results["num_tokens"]
            num_correct_tokens += op_results["num_correct_tokens"]

            print(
                "   Batch %4i: Epoch avg. loss: %.5f || Batch loss: %.5f | acc: %.5f"
                % (
                    step,
                    total_loss / num_samples,
                    op_results["loss"],
                    op_results["num_correct_tokens"]
                    / (float(op_results["num_tokens"]) + 1e-7),
                ),
                end="\r",
            )
        print("\r\x1b[K", end="")
        return (
            total_loss / num_samples,
            num_correct_tokens / float(num_tokens + 1e-7),
        )
