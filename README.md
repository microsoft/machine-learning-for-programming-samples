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
1. `Model.load_data_file` needs to be filled in to read a data file and a
   sequence of lists of tokens; each list is considered one sample.
   These could be all the tokens in one file, or all the tokens in one method.
   [This code should be reusable from the practical on the feature
    extractor]
2. `Model.load_metadata_from_dir` needs to be completed to compute a
   vocabulary from the data (use `load_data_file` to get the token
   sequences).
   
   To do this, the class `Vocabulary` from `dpu_utils.mlutils.vocabulary`
   can be used. `Vocabulary.create_vocabulary(...)` can be useful
   for this. 
3. `Model.load_data_from_dir` has to be completed to load the full data
   from disk into memory, tensorising it on the way.

   For each sequence of tokens produced by `load_data_file`, it should
   generate samples of some maximal length (e.g., 100), which are
   represented as int32 tensors (with each token represented by its
   corresponding vocabulary index).

   Each of the samples should be represented by a list of token ids in
   `loaded_data['tokens']` and the number of tokens in the sample in
   `loaded_data['tokens_lengths']`.

   `Vocabulary.get_id_or_unk_multiple()` can be useful for this step.
  
4. `Model.make_model` needs to be filled with the actual model, which
   should predict a token `tok[i]` based on the tokens `tok[:i]` seen
   so far.
   This should consume the placeholders defined in `Model.init_model` and
   produce a scalar `model.ops['loss']` that will be optimized.

   This method should consists of four steps:
   1. Create and use an embedding matrix used to map token IDs to a 
      distributed representation.
      `tf.nn.embedding_lookup` should be used in this subtask.
   2. Use an RNN to process the full sequence of tokens, producing
      one output per token.
      The utility function `tf.nn.dynamic_rnn` can be helpful here.
   3. Apply a transformation to map the RNN outputs to unnormalised
      probabilities over which token is the next one.
      This can be done by applying `tf.layers.dense` to the outputs
      of the RNN.
   4. Use `tf.nn.sparse_softmax_cross_entropy_with_logits` to compare
      the computed logits with the ground truth.
      Consider that if the input sequence is shorter than the maximum
      length allowed, predictions on the "padding" parts of the input
      should not contribute to the loss.
    
      After completing these steps, you should be able to train the model
      and observe the loss going down (the accuracy value will only be
      filled in after step 5):
```
$ python train.py trained_models/test/ train_data/ valid_data/
Loaded metadata for model:
           token_vocab: {'%PAD%': 0, '%UNK%': 1, 'self': 2, 'if': 3, 'return': 4, 'd
Training on 1031 samples.
Validating on 1031 samples.
==== Epoch 0 ====
  Epoch 0 (train) took 31.70s [processed 31 samples/second]
 Training Loss: 0.028717, Accuracy: 0.00%
  Epoch 0 (valid) took 9.14s [processed 109 samples/second]
 Validation Loss: 0.024548, Accuracy: 0.00%
  Best result so far -- saving model as 'trained_models/test/Model_RNNModel-2019-01-14-18-52-05_model_best.pkl.gz'.
[...]
```

5. `Model.make_model` should be extended to also compute the number
   of correct predictions, so that accuracy of the model can easily
   be computed. This part does not need to be differentiable, and so
   you can use `tf.argmax` to determine the most likely token at
   each step, and `tf.equal` to check that it is identical to the
   ground truth.
   The number of correctly predicted tokens in a minibatch should be
   exposed as `model.ops['num_correct_tokens']`.

   After completing this step, you should be able to test the model:
```
$ python test.py trained_models/test/Model_RNNModel-2019-01-14-18-52-05_model_best.pkl.gz test_data/
Test accuracy: 66.42%
```


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
