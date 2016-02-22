# Train a naiev dropout LSTM on a sentiment classification task.
# GPU command:
#     THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python script.py

# In[4]:

from __future__ import absolute_import
from __future__ import print_function

import sys
sys.path.insert(0, "/usr/local/cuda-7.0/bin")
sys.path.insert(0, "../keras") # point this to your local fork of https://github.com/yaringal/keras
sys.path.insert(0, "../Theano")
import theano
# Create ram disk: mount -t tmpfs -o size=512m tmpfs /mnt/ramdisk
# Use flag THEANO_FLAGS='base_compiledir=/mnt/ramdisk' python script.py
print('Theano version: ' + theano.__version__ + ', base compile dir: ' 
      + theano.config.base_compiledir)
theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

import numpy as np

from keras.preprocessing import sequence
from keras.preprocessing import text
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding, DropoutEmbedding
from keras.layers.recurrent import LSTM, GRU, DropoutLSTM, NaiveDropoutLSTM
from keras.callbacks import ModelCheckpoint, ModelTest
from keras.regularizers import l2
seed = 0


# In[5]:

if len(sys.argv) == 1:
  print("Expected args: p_W, p_U, p_dense, p_emb, weight_decay, batch_size, maxlen")
  print("Using default args:")
  sys.argv = ["", "0.5", "0.5", "0.5", "0.5", "1e-6", "128"]
args = [float(a) for a in sys.argv[1:]]
print(args)
p_W, p_U, p_dense, p_emb, weight_decay, batch_size = args
batch_size = int(batch_size)

nb_words = 20000
maxlen = 200  # cut texts after this number of words (among top max_features most common words)
start_char = 1
oov_char = 2
index_from = 3
skip_top = 0
test_split = 0.2

# In[6]:

print("Loading data...")
files = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]
texts, ratings = [], []
for file in files:
    with open("scale_data/scaledata/" + file + "/subj." + file, "r") as f:
        texts += list(f)
    with open("scale_data/scaledata/" + file + "/rating." + file, "r") as f:
        ratings += list(f)
tokenizer = text.Tokenizer(filters='')
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
Y = [float(rating) for rating in ratings]

np.random.seed(seed)
np.random.shuffle(X)
np.random.seed(seed)
np.random.shuffle(Y)

X = [[start_char] + [w + index_from for w in x] for x in X]

new_X = []
new_Y = []
for x, y in zip(X, Y):
#     if len(x) < maxlen:
#         new_X.append(x)
#         new_Y.append(y)
    for i in xrange(0, len(x), maxlen):
        new_X.append(x[i:i+maxlen])
        new_Y.append(y)
X = new_X
Y = new_Y

# by convention, use 2 as OOV word
# reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]

X_train = X[:int(len(X)*(1-test_split))]
Y_train = Y[:int(len(X)*(1-test_split))]
mean_y_train = np.mean(Y_train)
std_y_train = np.std(Y_train)
Y_train = [(y - mean_y_train) / std_y_train for y in Y_train]

X_test = X[int(len(X)*(1-test_split)):]
Y_test = Y[int(len(X)*(1-test_split)):]

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

# In[7]:

print("Pad sequences (samples x time)")
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)


# In[8]:

print('Build model...')
model = Sequential()
model.add(DropoutEmbedding(nb_words + index_from, 128, W_regularizer=l2(weight_decay), p=p_emb))
model.add(NaiveDropoutLSTM(128, 128, truncate_gradient=200, W_regularizer=l2(weight_decay), 
                      U_regularizer=l2(weight_decay), 
                      b_regularizer=l2(weight_decay), 
                      p_W=p_W, p_U=p_U))
model.add(Dropout(p_dense))
model.add(Dense(128, 1, W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay)))

#optimiser = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
optimiser = 'adam'
model.compile(loss='mean_squared_error', optimizer=optimiser)


# In[ ]:

# model.load_weights("/scratch/home/Projects/rnn_dropout/exps/DropoutLSTM_weights_00540.hdf5")


# In[ ]:

print("Train...")
# folder = "/scratch/home/Projects/rnn_dropout/exps/"
# checkpointer = ModelCheckpoint(filepath=folder+filename, 
#     verbose=1, append_epoch_name=True, save_every_X_epochs=10)
modeltest_1 = ModelTest(X_train[:100], mean_y_train + std_y_train * np.atleast_2d(Y_train[:100]).T, 
                      test_every_X_epochs=1, verbose=0, loss='euclidean',
                      mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
modeltest_2 = ModelTest(X_test, np.atleast_2d(Y_test).T, test_every_X_epochs=1, verbose=0, loss='euclidean',
                      mean_y_train=mean_y_train, std_y_train=std_y_train, tau=0.1)
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1250, 
          callbacks=[modeltest_1, modeltest_2]) #checkpointer, 
# score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
# print('Test score:', score)
# print('Test accuracy:', acc)


# In[ ]:

standard_prob = model.predict(X_train, batch_size=500, verbose=1)
print(np.mean(((mean_y_train + std_y_train * np.atleast_2d(Y_train).T) 
               - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)


# In[ ]:

standard_prob = model.predict(X_test, batch_size=500, verbose=1)
#print(standard_prob)
T = 50
prob = np.array([model.predict_stochastic(X_test, batch_size=500, verbose=0) 
                 for _ in xrange(T)])
prob_mean = np.mean(prob, 0)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * standard_prob))**2, 0)**0.5)
print(np.mean((np.atleast_2d(Y_test).T - (mean_y_train + std_y_train * prob_mean))**2, 0)**0.5)
