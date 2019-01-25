# Samples for Machine Learning for Programming

These are samples used in the University of Cambridge course 
[Machine Learning for Programming](https://www.cl.cam.ac.uk/teaching/1819/R252/).

## A Simple Language Model

Scaffolding for a simple TensorFlow language model is provided in
`language_model/`.
If you want to re-use this, install required dependencies using
`pip install -r requirements.txt`.
To turn this into a working model, five changes are required in `model.py`
(these are marked by `#TODO#'`):
1. `Model.load_data_file` needs to be filled in to read a data file and 
   return a sequence of lists of tokens; each list is considered one sample.
   These could be all the tokens in one file, or all the tokens in one method.
   [This code should be reusable from the practical on the feature
    extractor]

   It is common practice to normalise capitalization of tokens (as the embedding
   of `foo` and `Foo` should be similar). Make sure that `load_data_file`
   transforms all tokens to lower (or upper) case.

   You should be able to test this as follows:
   ```
   $ python test_step1.py data/r252-corpus-features/org/apache/lucene/analysis/miscellaneous/DuplicateByteSequenceSpotter.java.proto
   Loaded token sequences:
   ['public', 'duplicatebytesequencespotter', 'lparen', 'rparen', 'lbrace', 'this', 'dot', 'nodesallocatedbydepth', 'eq', 'new', 'int', 'lbracket', '4', 'rbracket', 'semi', 'this', 'dot', 'bytesallocated', 'eq', '0', 'semi', 'root', 'eq', 'new', 'roottreenode', 'lparen', 'lparen', 'byte', 'rparen', '1', 'comma', 'null', 'comma', '0', 'rparen', 'semi', 'rbrace']
   ['public', 'void', 'startnewsequence', 'lparen', 'rparen', 'lbrace', 'sequencebufferfilled', 'eq', 'false', 'semi', 'nextfreepos', 'eq', '0', 'semi', 'rbrace']
   ...
   ```
2. `Model.load_metadata_from_dir` needs to be completed to compute a
   vocabulary from the data (use `load_data_file` to get the token
   sequences).
   
   To do this, use the class `Vocabulary` from [`dpu_utils.mlutils.vocabulary`](https://github.com/Microsoft/dpu-utils/blob/master/dpu_utils/mlutils/vocabulary.py).
   `Vocabulary.create_vocabulary(...)` should be used to create the vocabulary
   with its second parameter `max_size` corresponding to the vocabulary size
   hyperparameter (e.g. 5000).
   The result should be stored as `self.metadata['token_vocab']` (this will
   be used in the scaffolding for step 5).

   You can test this step as follows:
   ```
   $ python test_step2.py r252-corpus-features/org/apache/lucene/analysis/miscellaneous/
   Loaded metadata for model:
              token_vocab: {'%PAD%': 0, '%UNK%': 1, 'lparen': 2, 'rparen': 3, 'semi': 4
   ```

3. `Model.load_data_from_raw_sample_sequences` has to be completed to
   turn sequences of tokens into tensorised data.

   For each passed sequence of tokens, it should generate samples of
   some maximal length (e.g., 50), which are represented as int32 tensors
   (with each token represented by its corresponding vocabulary index).

   Each of the samples should be represented by a list of token ids in
   `loaded_data['tokens']` and the number of tokens in the sample in
   `loaded_data['tokens_lengths']`.
   `Vocabulary.get_id_or_unk_multiple()` can be useful for this step.

   You can test this step as follows:
   ```
   $ python test_step3.py r252-corpus-features/org/apache/lucene/analysis/miscellaneous/ 
   Sample 0:
    Real length: 13
    Tensor length: 50
    Raw tensor: [ 21  22  15  29 147   2   3   8  77   6  11   4   9   0   0] (truncated)
    Interpreted tensor: ['monkeys_at', 'override', 'public', 'void', 'clear', 'lparen', 'rparen', 'lbrace', 'numpriorusesinasequence', 'eq', '0', 'semi', 'rbrace', '%PAD%', '%PAD%'] (truncated)
   ...
   ```
  
4. `Model.make_model` needs to be filled with the actual model, which
   should predict a token `tok[i]` based on the tokens `tok[:i]` seen
   so far.
   This should consume the placeholders defined in `Model.init` and
   produce a scalar `model.ops['loss']` that will be optimized.

   This method should consists of four steps:
   1. Create and use an embedding matrix used to map token IDs to a 
      distributed representation.
      [`tf.nn.embedding_lookup`](https://www.tensorflow.org/api_docs/python/tf/nn/embedding_lookup)
      should be used in this subtask.
      You might find [`tf.Variable`](https://www.tensorflow.org/api_docs/python/tf/Variable)
      and [`tf.random.uniform`](https://www.tensorflow.org/api_docs/python/tf/random/uniform)
      useful to populate the first parameter of
      `tf.nn.embedding_lookup`.
      An embedding dimension of 64 yields good results on our small
      dataset.
   2. Use an RNN to process the full sequence of tokens, producing
      one output per token.
      The utility function [`tf.nn.dynamic_rnn`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
      can be helpful here.
      As an RNN cell you might like to use [`tf.keras.layers.SimpleRNNCell`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNNCell).
      A hidden dimension of 64 yields good results on our dataset.
   3. Apply a transformation to map the RNN outputs to a vector of
      unnormalised probabilities (logits) that can be interpreted as
      a probability distribution over candidates for the next token.
      This can be done by applying [`tf.layers.dense`](https://www.tensorflow.org/api_docs/python/tf/layers/dense) to the outputs
      of the RNN.
   4. Use [`tf.nn.sparse_softmax_cross_entropy_with_logits`](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits) to compare
      the computed logits with the ground truth.

      **Note**: Consider that if the input sequence is shorter than the
      maximum length allowed, predictions on the "padding" parts of the
      input should not contribute to the loss. See task 5 for details.
    
    After completing these steps, you should be able to train the model
    and observe the loss going down (the accuracy value will only be
    filled in after step 6):
    ```
    $ python train.py trained_models/ data/r252-corpus-features/org/elasticsearch/{xpack,search}
    Loaded metadata for model:
              token_vocab: {'%PAD%': 0, '%UNK%': 1, 'lparen': 2, 'rparen': 3, 'dot': 4,
    Training on 57960 samples.
    Validating on 29441 samples.
    ==== Epoch 0 ====
      Epoch 0 (train) took 44.91s [processed 1290 samples/second]
    Training Loss: 0.023916, Accuracy: 13.63%
      Epoch 0 (valid) took 9.11s [processed 3230 samples/second]
    Validation Loss: 0.021208, Accuracy: 16.17%
      Best result so far -- saving model as 'trained_models/RNNModel-2019-01-22-16-56-50_model_best.pkl.gz'.
    ==== Epoch 1 ====
    [...]
    ```

5. To actually make predictions, you only need to extend `Model.make_model` to
   assign the unnormalised probabilities over the next token to
   `model.ops['output_logits']`. Once that is done, you should be able to test
   predictions:
    ```
    $ python predict.py trained_models/RNNModel-2019-01-22-16-56-50_model_best.pkl.gz "public"
    Prediction at step 0 (tokens ['public']):
     Prob 0.481: void
     Prob 0.085: static
     Prob 0.080: %UNK%
    Continuing with token void
    Prediction at step 1 (tokens ['public', 'void']):
     Prob 0.734: %UNK%
     Prob 0.020: lparen
     Prob 0.010: docheckwithstatuscode
    Continuing with token %UNK%
    ...
    ```
   **Note**: Note that tokens such as `{` and `(` are represented as 
    `lbrace` and `lparen` by the feature extractor and need to be used 
    the same way here, for example as follows:
    ```
    $ python predict.py trained_models/RNNModel-2019-01-22-16-56-50_model_best.pkl.gz public int foobar lparen string
    Prediction at step 0 (tokens ['public', 'int', 'foobar', 'lparen', 'string']):
    Prob 0.214: %UNK%
    Prob 0.034: ellipsis
    Prob 0.029: id
    Continuing with token %UNK%
    Prediction at step 1 (tokens ['public', 'int', 'foobar', 'lparen', 'string', '%UNK%']):
    Prob 0.503: comma
    Prob 0.473: rparen
    Prob 0.008: dot
    Continuing with token comma
    ...
    ```

6. Finally, `Model.make_model` should be extended to also compute the number
   of correct predictions, so that accuracy of the model can easily
   be computed. This part does not need to be differentiable, and so
   you can use [`tf.arg_max`](https://www.tensorflow.org/api_docs/python/tf/arg_max) to determine the most likely token at
   each step, and `tf.equal` to check that it is identical to the
   ground truth.
   The number of correctly predicted tokens in a minibatch should be
   exposed as `model.ops['num_correct_tokens']`.

   After completing this step, you should be able to evaluate the model:
    ```
    $ python evaluate.py trained_models/RNNModel-2019-01-22-16-56-50_model_best.pkl.gz data/r252-corpus-features/org/elasticsearch/common
      Epoch Test took 4.75s [processed 3041 samples/second]
    Test accuracy: 50.11%
    ```
   You may observe accuracies over 100%, which are most likely due to
   the fact that your number of correct tokens takes predictions of
   the `%PAD%` token into account. The next task should resolve this.

7. To improve training, we want to ignore those parts of the sequence that are
   just `%PAD%` symbols introduced to get to a uniform length. To this end,
   we need to mask out part of the loss (for tokens that are irrelevant).
   Such a 1.0/0.0 mask can be computed from the sequence lengths as follows:
   ```(python)
    # Step 1: Creates a tensor containing a list of token indices, i.e., [0, 1, 2, ..., T-1]
    index_list = tf.range(tf.shape(self.placeholders['tokens'])[1])  # Shape: [T]
    # Step 2: Replicate once for each entry in the batch, i.e., 
    #  [[0, 1, ..., T-1], ..., [0, 1, ..., T-1]]
    index_tensor = tf.tile(tf.expand_dims(index_list, axis=0),
                            multiples=(tf.shape(self.placeholders['tokens'])[0],
                                      1))  # Shape: [B, T]
    # Step 3: Turn into a 0/1 mask by comparing to the length of each batch entry, resulting in
    #  [[True, True, ..., True, False, ..., False], ...], which when cast to float gives us the
    #  required mask.
    loss_mask = tf.cast(index_tensor < tf.expand_dims(self.placeholders['tokens_lengths'], axis=1),
                        dtype=tf.float32)  # Shape: [B, T]
   ```
   You can use this to improve training as well as fix the computation
   of correct accuracy values.


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
