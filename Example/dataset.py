import numpy as np
from keras.preprocessing import sequence
from keras.preprocessing import text

class loader(object):
	def __init__(self, init_seed, maxlen, nb_words, skip_top, test_split):
		self.start_char = 1
		self.oov_char = 2
		self.index_from = 3

		files = ["Dennis+Schwartz", "James+Berardinelli", "Scott+Renshaw", "Steve+Rhodes"]
		texts, ratings = [], []
		for file in files:
		    with open("../Sentiment_analysis_code/scale_data/scaledata/" + file + "/subj." + file, "r") as f:
		        texts += list(f)
		    with open("../Sentiment_analysis_code/scale_data/scaledata/" + file + "/rating." + file, "r") as f:
		        ratings += list(f)
		tokenizer = text.Tokenizer(filters='')
		tokenizer.fit_on_texts(texts)
		X = tokenizer.texts_to_sequences(texts)
		Y = [float(rating) for rating in ratings]

		# Shuffle data:
		np.random.seed(init_seed)
		np.random.shuffle(X)
		np.random.seed(init_seed)
		np.random.shuffle(Y)

		# Parse data
		X = [[self.start_char] + [w + self.index_from for w in x] for x in X]

		new_X = []
		new_Y = []
		for x, y in zip(X, Y):
		    for i in xrange(0, len(x), maxlen):
		        new_X.append(x[i:i+maxlen])
		        new_Y.append(y)
		X = new_X
		Y = new_Y

		# by convention, use 2 as OOV word
		# reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
		X = [[self.oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]

		self.X_train = X[:int(len(X)*(1-test_split))]
		self.Y_train = Y[:int(len(X)*(1-test_split))]
		self.mean_y_train = np.mean(self.Y_train)
		self.std_y_train = np.std(self.Y_train)
		self.Y_train = [(y - self.mean_y_train) / self.std_y_train for y in self.Y_train]

		self.X_test = X[int(len(X)*(1-test_split)):]
		self.Y_test = Y[int(len(X)*(1-test_split)):]

		print(len(self.X_train), 'train sequences')
		print(len(self.X_test), 'test sequences')

		print("Pad sequences (samples x time)")
		self.X_train = sequence.pad_sequences(self.X_train, maxlen=maxlen)
		self.X_test = sequence.pad_sequences(self.X_test, maxlen=maxlen)
		print('X_train shape:', self.X_train.shape)
		print('X_test shape:', self.X_test.shape)
