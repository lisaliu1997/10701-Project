import numpy as np
from np import array
import keras
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
import keras.backend as K 

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
	train_inputs = Input(shape=(None, n_input))

	# Add LSTM layer
	train_lstm = LSTM(n_units, return_sequences=True)
	train_outputs = train_lstm(train_inputs)

    # Add dense layer
	train_dense = Dense(n_features, activation='sigmoid')
	train_outputs = train_dense(train_outputs)

	# Define training model
	train_model = Model([train_inputs], train_outputs)

	return train_model

def build_train(X, Y, n_features, n_units):
	def hybrid_loss(y_true, y_pred):
		mse = mean_squared_error(y_true, y_pred)
		corr = correlation_coefficient_loss(y_true, y_pred)
		knob = config["loss_regulizer"]
		loss = corr + knob * mse
		return loss
	train_model = lstm_model(n_features, n_units)
	train_model.compile(optimizer='adam', loss=hybrid_loss, metrics=[correlation_coefficient_loss, 'mse'])
	train_model.fit(X, Y, epochs=100, verbose=1, validation_split=0.1)

	return train_model


def predict_sequence(model, n_steps, n_features):
	batch_size = (source.shape)[0]
	# start of sequence input
	target_seq = array([0.0 for _ in range(n_features)]).reshape(1, 1, n_features)
	# collect predictions
	output = list()
	for t in range(n_steps):
		yhat = model.predict([target_seq])
		output.append(yhat[0, 0, :])
		target_seq = yhat

	output = array(output).flatten()
	return output

