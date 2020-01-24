# Samples for Machine Learning for Programming

These are samples used in the University of Cambridge course 
[Machine Learning for Programming](https://www.cl.cam.ac.uk/teaching/1920/P252/).

## A Simple Language Model

Scaffolding for a simple language model is provided in `language_model/`, for
TensorFlow 1.X, TensorFlow 2.X, and PyTorch. Python 3.6 or later is required.
If you want to re-use this, pick a framework you want to use, install it and
the requirements for this model using `pip install -r requirements.txt`.

To get started, open a console and change your current directory to `language_model/`.
Alternatively, set that directory to your `PYTHONPATH` enviornment variable:
```
export PYTHONPATH=/path/to/language_model
```


The scaffold provides some generic code to simplify the task (such as a 
training loop, logic for saving and restoring, ...), but you need to complete 
the code in a number of places to obtain a working model (these are marked by 
`#TODO N#` in the code):
1. In `model.py`, uncomment the line corresponding to the framework you want to
   use.

2. In `dataset.py`, `load_data_file` needs to be filled in to read a data file
   and return a sequence of lists of tokens; each list is considered one 
   sample.
   This should re-use the code from the first practical to provide one sample
   for the tokens in each method.

   It is common practice to normalise capitalization of tokens (as the
   embedding of `foo` and `Foo` should be similar). Make sure that 
   `load_data_file` transforms all tokens to lower (or upper) case.

   You should be able to test this as follows:
   ```
   $ python test_step2.py data/jsoup/src/main/java/org/jsoup/Jsoup.java.proto | tail -n -1
   ['public', 'static', 'boolean', 'isvalid', 'lparen', 'string', 'bodyhtml', 'comma', 'whitelist', 'whitelist', 'rparen', 'lbrace', 'return', 'new', 'cleaner', 'lparen', 'whitelist', 'rparen', 'dot', 'isvalidbodyhtml', 'lparen', 'bodyhtml', 'rparen', 'semi', 'rbrace']
   ```

3. In `dataset.py`, `build_vocab_from_data_dir` needs to be completed to 
   compute a vocabulary from the data.
   The vocabulary will be used to represent all tokens by integer IDs, and
   we need to consider three special tokens: the `UNK` token used to represent
   infrequent tokens and those not seen at training time, the `PAD` token used
   to make all samples of the same length, and `START_SYMBOL` token used to
   as the first token in every sample and the `END_SYMBOL` used as the last.

   To do this, we use the class `Vocabulary` from [`dpu_utils.mlutils.vocabulary`](https://github.com/Microsoft/dpu-utils/blob/master/python/dpu_utils/mlutils/vocabulary.py).
   Using `load_data_file` from above, compute the frequency of tokens in the
   passed `data_dir` (`collections.Counter` is useful here) and use that
   information to add the `vocab_size` most common of them to `vocab`.

   You can test this step as follows:
   ```
   $ python test_step3.py data/jsoup/src/main/java/org/jsoup/
   Loaded vocabulary for dataset:
   {'%PAD%': 0, '%UNK%': 1, '%START%': 2, '%END%': 3, 'rparen': 4, 'lparen': 5, 'semi': 6, 'dot': 7, 'rbrace': 8, ' [...]
   ```

4. In `dataset.py`, `tensorise_token_sequence` needs to be completed to 
   translate a token sequence into a sequence of integer token IDs of
   uniform length.
   
   The output of the function should always be a list of length `length`
   of token IDs from `vocab`, where longer sequences are truncated and shorter
   sequences are padded to the correct length.
   We also want to use this method to insert the `START_SYMBOL` at the
   beginning of each sample. The special `END_SYMBOL` symbol needs to be appended
   to indicate the end of a list of tokens, whereas a special `PAD_SYMBOL` needs
   to be added to serve as a filler so that all token sequences will have the same length.

   You can test this step as follows: (note this is an example output that is using count_threshold of 2)
   ```
   $ python test_step4.py data/jsoup/src/main/java/org/jsoup/
   Sample 0:
   Real length: 50
   Tensor length: 50
   Raw tensor: [  2  13   1   4   3   8 118   4   3   5   7  13   1   4  12   1   3   8
   118   4   1   3   5   7  13   1   4   1   1   3   8 118   4   1   3   5
      7  13   1   4  12   1   9   1   1   3   8 118   4   1] (truncated)
   Interpreted tensor: ['%START%', 'public', '%UNK%', 'lparen', 'rparen', 'lbrace', 'super', 'lparen', 'rparen', 'semi', 'rbrace', 'public', '%UNK%', 'lparen', 'string', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%', 'rparen', 'semi', 'rbrace', 'public', '%UNK%', 'lparen', '%UNK%', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%', 'rparen', 'semi', 'rbrace', 'public', '%UNK%', 'lparen', 'string', '%UNK%', 'comma', '%UNK%', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%'] (truncated)
   Sample 1:
   Real length: 46
   Tensor length: 50
   Raw tensor: [  2  13   1   4  12   1   3   8 118   4   1   3   5   7  13   1   4   1
      1   3   8 118   4   1   3   5   7  13   1   4  12   1   9   1   1   3
      8 118   4   1   9   1   3   5   7   7   0   0] (truncated)
   Interpreted tensor: ['%START%', 'public', '%UNK%', 'lparen', 'string', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%', 'rparen', 'semi', 'rbrace', 'public', '%UNK%', 'lparen', '%UNK%', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%', 'rparen', 'semi', 'rbrace', 'public', '%UNK%', 'lparen', 'string', '%UNK%', 'comma', '%UNK%', '%UNK%', 'rparen', 'lbrace', 'super', 'lparen', '%UNK%', 'comma', '%UNK%', 'rparen', 'semi', 'rbrace', 'rbrace', '%PAD%', '%PAD%'] (truncated)
   ...
   ```

5. The actual model needs to be built.
   Our goal is to learn to predict `tok[i]` based on the token `tok[:i]` seen
   so far.
   The process and scaffold is very similar in all frameworks. The
   method `compute_logits` and `compute_loss_and_acc` need to be completed,
   and the `build` method can always be used to initialise weights and
   layers that will be re-used during training and prediction.
   Parameters such as `EmbeddingDim` and `RNNDim` should be hyperparameters,
   but values such as `64` work well.

   1) In `compute_logits`, implement the logic to embed the `token_ids` input
      tensor into a distributed representation.
      In TF 1.x, you can use [`tf.nn.embedding_lookup`](https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/embedding_lookup);
      in TF 2.X, you can use [`tf.keras.layers.Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding);
      and in PyTorch, you can use [`torch.nn.Embedding`](https://pytorch.org/docs/master/nn.html#torch.nn.Embedding) for this purpose.

      This should translate an `int32` tensor of shape `[Batch, Timesteps]`
      into a `float32` tensor of shape `[Batch, Timesteps, EmbeddingDim]`.
   
   2) In `compute_logits`, implement an actual RNN consuming the results of
      the embedding layer. You can use [`tf.keras.layers.GRU`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRU)
      resp. [`torch.nn.GRU`](https://pytorch.org/docs/master/nn.html#torch.nn.GRU)
      (or their LSTM variants) for this.
      This should translate a `float32` tensor of shape `[Batch, Timesteps,
      EmbeddingDim]` into a `float32` tensor of shape `[Batch, Timesteps, 
      RNNDim]`.
   
   3) In `compute_logits`, implement a linear layer to translate the RNN
      output into an unnormalised probability distribution over the the
      vocabulary. You can use [`tf.keras.layers.Dense`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense) 
      resp. [`torch.nn.Linear`](https://pytorch.org/docs/master/nn.html#torch.nn.Linear)
      for this.
      This should translate a `float32` tensor of shape `[Batch, Timesteps,
      RNNDim]` into a `float32` tensor of shape `[Batch, Timesteps, 
      VocabSize]`.

   4) In `compute_loss_and_acc`, implement a cross-entropy loss that compares
      the probability distribution computed at timestep `T` with the input
      at timestep `T+1` (which is the token that we want to predict).
      Note that this means that we need to discard the final RNN output, as we
      do not know the next token.
      You can use [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits) resp.
      [`torch.nn.CrossEntropyLoss`](https://pytorch.org/docs/master/nn.html?highlight=crossentropyloss#torch.nn.CrossEntropyLoss) for this.

   After completing these steps, you should be able to train the model
   and observe the loss going down (the accuracy value will only be
   filled in after step 6):
   ```
   $ python train.py trained_models data/jsoup/{,}
   Loading data ...
     Built vocabulary of 4697 entries.
     Loaded 2233 training samples from data/jsoup/.
     Loaded 2233 validation samples from data/jsoup/.
   Running model on GPU.
   Constructed model, using the following hyperparameters: {"optimizer": "Adam", "learning_rate": 0.01, "learning_rate_decay": 0.98, "momentum": 0.85, "max_epochs": 500, "patience": 5, "max_vocab_size": 10000, "max_seq_length": 50, "batch_size": 200, "token_embedding_size": 64, "rnn_type": "GRU", "rnn_num_layers": 2, "rnn_hidden_dim": 64, "rnn_dropout": 0.2, "use_gpu": true, "run_id": "RNNModel-2019-12-29-13-23-18"}
   Initial valid loss: 0.042.
   [...]
   == Epoch 1
    Train:  Loss 0.0303, Acc 0.000
    Valid:  Loss 0.0224, Acc 0.000
     (Best epoch so far, loss decreased 0.0224 from 0.0423)
     (Saved model to trained_models/RNNModel-2019-12-29-13-23-18_best_model.bin)
   == Epoch 2
    Train:  Loss 0.0213, Acc 0.000
    Valid:  Loss 0.0195, Acc 0.000
     (Best epoch so far, loss decreased 0.0195 from 0.0224)
     (Saved model to trained_models/RNNModel-2019-12-29-13-23-18_best_model.bin)
   [...]
   ```

   The saved models should already be usable as autocompletion models, using
   the provided `predict.py` script:
   ```
   $ python predict.py trained_models/RNNModel-2019-12-29-13-23-18_best_model.bin public
   Prediction at step 0 (tokens ['public']):
    Prob 0.282: static
    Prob 0.099: void
    Prob 0.067: string
   Continuing with token static
   Prediction at step 1 (tokens ['public', 'static']):
    Prob 0.345: void
    Prob 0.173: document
    Prob 0.123: string
   Continuing with token void
   Prediction at step 2 (tokens ['public', 'static', 'void']):
    Prob 0.301: main
    Prob 0.104: isfalse
    Prob 0.089: nonullelements
   Continuing with token main
   Prediction at step 3 (tokens ['public', 'static', 'void', 'main']):
    Prob 0.999: lparen
    Prob 0.000: filterout
    Prob 0.000: iterator
   Continuing with token lparen
   Prediction at step 4 (tokens ['public', 'static', 'void', 'main', 'lparen']):
    Prob 0.886: string
    Prob 0.033: int
    Prob 0.030: object
   Continuing with token string
   ```
   **Note**: Note that tokens such as `{` and `(` are represented as 
    `lbrace` and `lparen` by the feature extractor and are used 
    the same way here.

6. Finally, `compute_loss_and_acc` should be extended to also compute the
   number of (correct) predictions, so that accuracy of the model can be
   computed.
   For this, you need to check if the most likely prediction corresponds to
   the ground truth. You can use `tf.argmax` resp. `torch.argmax` here.
   Finally, we also need to discount padding tokens, so you need to compute
   a mask which predictions correspond to padding. Here, you can use
   `self.vocab.get_id_or_unk(self.vocab.get_pad())` to get the integer ID
   of the padding token.

   After completing this step, you should be able to evaluate the model:
   ```
   $ python evaluate.py trained_models/RNNModel-2019-12-29-13-23-18_best_model.bin data/jsoup/
   Loading data ...
     Loaded trained model from trained_models/RNNModel-2019-12-29-13-23-18_best_model.bin.
     Loaded 2233 test samples from data/jsoup/.
   Test:  Loss 24.9771, Acc 0.876
   ```

7. To improve training, we want to ignore those parts of the sequence that are
   just `%PAD%` symbols introduced to get to a uniform length. To this end,
   we need to mask out part of the loss (for tokens that are irrelevant).
   You can use the mask computed in step 6 again here.


# Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
