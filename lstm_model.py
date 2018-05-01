import numpy as np
import keras
from keras import losses
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
import keras.backend as K 
from data_processing import new_reconstruct_ifft
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(xm*ym)
    r_den = K.sqrt((K.sum(K.square(xm)))*(K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def lstm_model(n_features, n_units):
	model = Sequential()
	model.add(TimeDistributed(Dense(input_dim=n_features, output_dim=n_units)))
	model.add(LSTM(input_dim=n_units, output_dim=n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(input_dim=n_units, output_dim=n_features)))

	return model

def plot_metric(history, key):
	if (key.startswith('val')): return
	plt.plot(history.history[key])
	plt.plot(history.history['val_%s'%key])
	plt.title('model %s'%key)
	plt.ylabel(key)
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper right')
	plt.savefig("plots/lstm_fixed_%s.png"%key)
	plt.clf()

def build_train(X, Y, n_features, n_units):
	X_flipped = np.flip(X, 1)
	def hybrid_loss(y_true, y_pred):
		mse = losses.mean_squared_error(y_true, y_pred)
		corr = correlation_coefficient_loss(y_true, y_pred)
		knob = 5000
		loss = corr + knob * mse
		return loss
	# train_model = lstm_model(n_features, n_units)
	# adam = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# train_model.compile(optimizer='adam', loss=hybrid_loss, metrics=[correlation_coefficient_loss,'mse'])
	model = lstm_model(n_features, n_units)
	model.compile(loss='mse', optimizer='rmsprop')
	history = model.fit(X, Y, batch_size=1, epochs=20, verbose=1, validation_split=0.1)
	print(history.history.keys())
	for key in history.history.keys():
		plot_metric(history, key)

	return model

def predict_sequence(model, seed, n_steps, n_features, n_units):
	n_steps = 10
	print("Predict sequence of length %d" %n_steps)
	# start of sequence input
	target_seq = seed.copy()
	# collect predictions
	output = list()
	for t in range(n_steps):
		# print("t=%d" %t)
		yhat = model.predict([target_seq])
		# store prediction
		new = yhat[0][-1].copy()
		# print(new[:5])
		output.append(new)
		# update target sequence
		new = np.reshape(new, (1, 1, new.shape[0]))
		target_seq = np.concatenate((target_seq, new), axis=1)
		target_seq = target_seq[:, 1:, :]
		if (t % 500 == 0):
			print("t=%d" %t)

	output = np.array(output)
	new_reconstruct_ifft.convert_back_wav("results/lstm_fixed_pred.wav", output)
	print("Outputs first")
	print(output[:10, :5])
	print("Outputs later")
	print(output[200:210, :5])
	# output is a 2D array
	


