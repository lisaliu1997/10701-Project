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
from keras.models import Sequential, Model
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

# def lstm_model(n_features, n_units):
# 	train_inputs = Input(shape=(None, n_features))

# 	# Add LSTM layer
# 	train_lstm = LSTM(n_units, return_sequences=True)
# 	train_outputs = train_lstm(train_inputs)

#     # Add dense layer
# 	train_dense = Dense(n_features, activation='sigmoid')
# 	train_outputs = train_dense(train_outputs)

# 	# Define training model
# 	train_model = Model([train_inputs], train_outputs)

# 	return train_model

def new_lstm_model(n_features, n_units):
	model = Sequential()
	model.add(TimeDistributed(Dense(n_units), input_shape=(None, n_features)))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(n_features)))

	return model

# def define_model(n_features, n_units):
# 	inputs = Input(shape=(None, n_features))

# 	# Define layers
# 	lstm_layer = LSTM(n_units, return_sequences=True, return_state=True)
# 	dense_layer = Dense(n_features, activation='sigmoid')

# 	# Training model
# 	outputs, _, _ = lstm_layer(inputs)
# 	outputs = dense_layer(outputs)
# 	train_model = Model(inputs, outputs)

# 	# Inference model
# 	state_input_h = Input(shape=(n_units,))
# 	state_input_c = Input(shape=(n_units,))
# 	states_inputs = [state_input_h, state_input_c]
# 	outputs, state_h, state_c = lstm_layer(inputs, initial_state=states_inputs)
# 	states = [state_h, state_c]
# 	outputs = dense_layer(outputs)
# 	inf_model = Model([inputs] + states_inputs, [outputs] + states)

# 	return train_model, inf_model

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
	model = new_lstm_model(n_features, n_units)
	model.compile(loss='mse', optimizer='rmsprop')

	history = model.fit(X, Y, epochs=100, verbose=1, validation_split=0.1)
	print(history.history.keys())
	for key in history.history.keys():
		plot_metric(history, key)

	model.save_weights('weights/my_model_weights.h5')
	model.load_weights('weights/my_model_weights.h5')

	return model

# def predict_sequence(model, seed, n_steps, n_features, n_units):
# 	print("Predict sequence of length %d" %n_steps)
# 	# Initialize states
# 	h = np.zeros((1, n_units))
# 	c = np.zeros((1, n_units))
# 	state = [h, c]
# 	print("Initial states:")
# 	print("h:")
# 	print(h[0, :5])
# 	print("c:")
# 	print(c[0, :5])
# 	# start of sequence input
# 	target_seq = seed.copy()
# 	# collect predictions
# 	output = list()
# 	for t in range(n_steps):
# 		# predict next step
# 		yhat, h, c = model.predict([target_seq] + state)
# 		new = yhat[0, 0, :]
# 		# print("yhat shape", yhat.shape)
# 		# store prediction
# 		output.append(new) 
# 		# update state
# 		state = [h, c]
# 		# update target sequence
# 		target_seq = new.reshape((1, 1, new.shape[0]))
# 		# target_seq = np.random.uniform(0, 1, (1, 1, new.shape[0]))
# 		if (t <= 10):
# 			print("h:", h[0, :5])
# 			print("c:", c[0, :5])	
# 		if (t % 500 == 0):
# 			print("t=%d" %t)
# 			print("h:", h[0, :5])
# 			print("c:", c[0, :5])	
# 	output = np.array(output)
# 	print("Outputs")
# 	print(output[:10, :5])
# 	print("Outputs")
# 	print(output[3000:3010, :5])
# 	# output is a 2D array
# 	new_reconstruct_ifft.convert_back_wav("results/prediction.wav", output)

def predict_sequence(model, seed, n_steps, n_features, filename):
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
		# print("t=%d"%t)
		# print(new[:5])
		# print(new[:5])
		output.append(new)
		# update target sequence
		new = np.reshape(new, (1, 1, new.shape[0]))
		target_seq = np.concatenate((target_seq, new), axis=1)
		# Discard the first time step s.t. target_seq always have fixed length.
		target_seq = target_seq[:, 1:, :]
		if (t % 500 == 0):
			print("t=%d" %t)

	output = np.array(output)
	print("prediction")
	print(output[:100, :5])
	new_reconstruct_ifft.convert_back_wav("results/%s.wav"%filename, output)
	# print("Outputs first")
	# print(output[:10, :5])
	# print("Outputs later")
	# print(output[200:210, :5])
	# output is a 2D array
	



