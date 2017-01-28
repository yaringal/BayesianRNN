This is the code used for the experiments in the paper ["A Theoretically Grounded Application of Dropout in Recurrent Neural Networks"](http://mlg.eng.cam.ac.uk/yarin/publications.html#Gal2015Theoretically). The [sentiment analysis experiment](Sentiment_analysis_code/) relies on a [fork of keras](https://github.com/yaringal/keras/tree/BayesianRNN) which implements Bayesian LSTM, Bayesian GRU, embedding dropout, and MC dropout. The [language model experiment](LM_code/) extends [wojzaremba's lua code](https://github.com/wojzaremba/lstm).

## Update 1 (Feb 22): 
[Keras](https://keras.io/layers/recurrent/) now supports dropout in RNNs following the implementation above. A simplified example of the [sentiment analysis experiment](Sentiment_analysis_code/) using the latest keras implementation is given in [here](Example/).

## Update 2 (March 28): 
The script [main_new_dropout_SOTA](LM_code/main_new_dropout_SOTA.lua) implements Bayesian LSTM (Gal, 2015) for the large model of Zaremba et al. (2014). In the setting of Zaremba et al. the states are not reset and the testing is done with a single pass through the test set. The only changes I've made to the setting of Zaremba et al. are:

1. dropout technique (using a Bayesian LSTM)
2. weight decay (which was chosen to be zero in Zaremba et al.)
3. a slightly smaller network was used to fit on my GPU (1250 units per layer instead of 1500)

All other hypers being identical to Zaremba et al.: learning rate decay was not tuned for my setting and is used following Zaremba et al., and the sequences are initialised with the previous state following Zaremba et al. (unlike in main_dropout.lua). Dropout parameters were optimised with grid search (tying dropout_x & dropout_h and dropout_i & dropout_o) over validation perplexity (optimal values are 0.3 and 0.5 compared Zaremba et al.'s 0.6).

Single model validation perplexity is improved from Zaremba et al.'s 82.2 to 79.1. Test perplexity is reduced from 78.4 to 76.5, see [log](LM_code/main_new_dropout_SOTA.log). Evaluating the model with MC dropout with 2000 samples, test perplexity is further reduced to 75.06 (with 100 samples test perplexity is 75.3).

## Update 3 (July 6): 
I updated the code with the experiments used in the arXiv paper revision from 25 May 2016 ([version 3](http://arxiv.org/abs/1512.05287v3)).
In the updated code restriction 3 above (smaller network size) was removed, following a Lua update that solved a memory leak.
[main_new_dropout_SOTA_v3](LM_code/main_new_dropout_SOTA_v3.lua) implements the MC dropout experiment used in the paper, with **single model test perplexity improved from Zaremba et al.'s 78.4 to 73.4 (using MC dropout at test time) and 75.2 with the dropout approximation**. Validation perplexity is reduced from 82.2 to 77.9.

## Update 4 (January 1): 
I updated the script `main_new_dropout_SOTA_v3.lua` fixing a bug that @helson73 found (issue #4). In the original script, word embedding dropout was erroneously sampled anew for each word token (ie the word token masks were not tied in the LM experiment, unlike the sentiment analysis experiment). I fixed the code and re-ran the experiments with `Variational (untied weights) large LSTM`, giving a small improvement in perplexity:

- Validation set perplexity: 77.457 (down from 77.9)
- Test set perplexity: 75.112 (down from 75.2)
- Test set perplexity (MC): 73.318975131 (down from 73.4)

The improvement is rather small because the sequence length in the LM exps is 20. This means that most sequences will have unique words (ie a word would not appear multiple times in the sequence), hence having the masks untied in such sequences is the same as having the masks tied. Note that in longer sequences such as in the sentiment analysis exps (with sequence length of 200) most sequences will have common words (such as stop words) appearing multiple times in the sequence.

References:

* Gal, Y, "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks", 2015.
* Zaremba, W, Sutskever, I, Vinyals, O, "Recurrent neural network regularization", 2014.