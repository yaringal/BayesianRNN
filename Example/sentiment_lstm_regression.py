# Train a Bayesian LSTM on the IMDB sentiment classification task.
# To use the GPU:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm_regression.py
# To speed up Theano, create a ram disk: 
#     mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Then add flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk'

from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import sys
import theano
from callbacks import ModelTest
from dataset import loader
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2

# Process inpus:
if len(sys.argv) == 1:
  print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen")
  print("Using default args:")
  # sys.argv = ["", "0.", "0.", "0.", "0.", "1e-4", "128", "200"]
  sys.argv = ["", "0.25", "0.25", "0.25", "0.25", "1e-4", "128", "200"]
args = [float(a) for a in sys.argv[1:]]
print(args)
p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen = args
batch_size = int(batch_size)
maxlen = int(maxlen)
folder = "/scratch/home/Projects/rnn_dropout/exps/"
filename = ("sa_DropoutLSTM_pW_%.2f_pU_%.2f_pDense_%.2f_pEmb_%.2f_reg_%f_batch_size_%d_cutoff_%d_epochs"
  % (p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen))
print(filename)

# Global params:
nb_words = 20000
skip_top = 0
test_split = 0.2
init_seed = 0
global_seed = 0

# Load data:
print("Loading data...")
dataset = loader(init_seed, maxlen, nb_words, skip_top, test_split)
X_train, X_test, Y_train, Y_test = dataset.X_train, dataset.X_test, dataset.Y_train, dataset.Y_test
mean_y_train, std_y_train = dataset.mean_y_train, dataset.std_y_train

X_train = np.asarray(X_train)
X_test  = np.asarray(X_test)
Y_train = np.asarray(Y_train)
Y_test  = np.asarray(Y_test)

# Set seed:
np.random.seed(global_seed)

# Build model:
print('Build model...')
model = Sequential()
model.add(Embedding(nb_words + dataset.index_from, 128, W_regularizer=l2(weight_decay),
                    dropout=p_emb, input_length=maxlen, batch_input_shape=(batch_size, maxlen)))
model.add(LSTM(128, W_regularizer=l2(weight_decay), U_regularizer=l2(weight_decay),
               b_regularizer=l2(weight_decay), dropout_W=p_W, dropout_U=p_U))
model.add(Dropout(p_dense))
model.add(Dense(1, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

#optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss='mean_squared_error', optimizer=optimiser)

# Potentially load weights
# model.load_weights("path")

# Train model
print("Train...")

# Theano
modeltest_1 = ModelTest(X_train[:100], 
                        mean_y_train + std_y_train * np.atleast_2d(Y_train[:100]).T, 
                        test_every_X_epochs=1, verbose=0, loss='euclidean', 
                        mean_y_train=mean_y_train, std_y_train=std_y_train)
modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1, 
                        verbose=0, loss='euclidean', 
                        mean_y_train=mean_y_train, std_y_train=std_y_train)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=250, 
          callbacks=[modeltest_1, modeltest_2])

# # Tensorflow
# modeltest_1 = ModelTest(X_train[:batch_size],
#                         mean_y_train + std_y_train * np.atleast_2d(Y_train[:batch_size]).T, 
#                         test_every_X_epochs=1, verbose=0, loss='euclidean', 
#                         mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=batch_size)
# tensorflow_test_size = batch_size * (len(X_test) / batch_size)
# modeltest_2 = ModelTest(X_test[:tensorflow_test_size], np.atleast_2d(Y_test[:tensorflow_test_size]).T, 
#                         test_every_X_epochs=1, verbose=0, loss='euclidean', 
#                         mean_y_train=mean_y_train, std_y_train=std_y_train, batch_size=batch_size)
# tensorflow_train_size = batch_size * (len(X_train) / batch_size)
# model.fit(X_train[:tensorflow_train_size], Y_train[:tensorflow_train_size],
#           batch_size=batch_size, nb_epoch=250, callbacks=[modeltest_1, modeltest_2])

# Potentially save weights
# model.save_weights("path", overwrite=True)

# Evaluate model
# Dropout approximation for training data:
standard_prob = model.predict(X_train, batch_size=500, verbose=1)
print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T)
               - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)


# Dropout approximation for test data:
standard_prob = model.predict(X_test, batch_size=500, verbose=1)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)

# MC dropout for test data:
T = 50
prob = np.array([modeltest_2.predict_stochastic(X_test, batch_size=500, verbose=0)
                 for _ in xrange(T)])
prob_mean = np.mean(prob, 0)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * prob_mean))**2, 0)**0.5)
