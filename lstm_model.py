import numpy as np
import keras
from keras import losses
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import keras.backend as K 
from data_processing import reconstruct_ifft

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
	train_inputs = Input(shape=(None, n_features))

	# Add LSTM layer
	train_lstm = LSTM(n_units, return_sequences=True)
	train_outputs = train_lstm(train_inputs)

    # Add dense layer
	train_dense = Dense(n_features, activation='sigmoid')
	train_outputs = train_dense(train_outputs)

	# Define training model
	train_model = Model([train_inputs], train_outputs)

	return train_model

def define_model(n_features, n_units):
	inputs = Input(shape=(None, n_features))

	# Define layers
	lstm_layer = LSTM(n_units, return_sequences=True, return_state=True)
	dense_layer = Dense(n_features, activation='sigmoid')

	# Training model
	outputs, _, _ = lstm_layer(inputs)
	outputs = dense_layer(outputs)
	train_model = Model(inputs, outputs)

	# Inference model
	# state_input_h = Input(shape=(n_units,))
	# state_input_c = Input(shape=(n_units,))
	# states_inputs = [state_input_h, state_input_c]
	# outputs, state_h, state_c = lstm_layer(inputs, initial_state=states_inputs)
	# states = [state_h, state_c]
	# outputs = dense_layer(outputs)
	# inf_model = Model([inputs] + states_inputs, [outputs] + states)

	return train_model

def build_train(X, Y, n_features, n_units):
	def hybrid_loss(y_true, y_pred):
		mse = losses.mean_squared_error(y_true, y_pred)
		corr = correlation_coefficient_loss(y_true, y_pred)
		knob = 10
		loss = corr + knob * mse
		return loss
	train_model = define_model(n_features, n_units)
	train_model.compile(optimizer='adam', loss=hybrid_loss, metrics=[correlation_coefficient_loss,'mse'])
	train_model.fit(X, Y, epochs=2, verbose=1, validation_split=0.1)

	return train_model

def predict_sequence(model, seed, n_steps, n_features):
	print("Predict sequence of length %d" %n_steps)
	# start of sequence input
	target_seq = seed.copy()
	# collect predictions
	output = list()
	for t in range(n_steps):
		yhat = model.predict([target_seq])
		# store prediction
		if (t == 0):
			for i in range(yhat.shape[1]):
				output.append(yhat[0][i].copy())
		else:
			output.append(yhat[0][-1].copy())
		# update target sequence
		new = yhat[0][-1]
		new = np.reshape(new, (1, 1, new.shape[0]))
		target_seq = np.concatenate((target_seq, new), axis=1)
		if (t % 500 == 0):
			print("t=%d" %t)

	# output is a 2D array
	audio_data = reconstruct_ifft.reconstruct(output)
 	return audio_data

# def predict_sequence(model, seed, state, n_steps, n_features):
# 	print("Predict sequence of length %d" %n_steps)
# 	# start of sequence input
# 	target_seq = seed.copy()
# 	# collect predictions
# 	output = list()
# 	for t in range(n_steps):
# 		# predict next step
# 		yhat, h, c = model.predict([target_seq] + state)
# 		# store prediction
# 		output.append(yhat[0, 0, :])
# 		# update state
# 		state = [h, c]
# 		# update target sequence
# 		target_seq = yhat
# 		if (t % 500 == 0):
# 			print("t=%d" %t)

# 	# output is a 2D array
# 	audio_data = reconstruct_ifft.reconstruct(output)
# 	return audio_data




