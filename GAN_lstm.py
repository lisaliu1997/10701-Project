from __future__ import print_function, division

import keras
from keras import losses
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K 

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import sys
import os
import random
import numpy as np

# Import other .py files
from data_processing import new_reconstruct_ifft


# Possible changes: 
# 1. optimizers
# 2. seed (or noise)
# Note: 
# 1. All audio_data should be normalized to [0, 1]
# 2. Should discriminator look at raw data or spectrogram?

data_root = "data"
spect_folder = 'spectrogram'
max_n_files = 10

def get_data():
	input_files = os.listdir("%s/%s/"%(data_root, input_folder))
	X = np.zeros((1, time_steps, n_features))
	Y = np.zeros((1, time_steps, n_features))
	n_files = 0
	for input_file in input_files:
		if (not input_file.endswith("x.npy")): continue
		length = len(input_file)
		music_name = input_file[:(length-5)]
		print(music_name)
		x = np.load("%s/%s/%s"%(data_root, input_folder, input_file))
		output_file = "%sy.npy"%music_name
		output_file_path = "%s/%s/%s"%(data_root, output_folder, output_file)
		if (not os.path.isfile(output_file_path)): continue
		y = np.load(output_file_path)
		print("x: ", x.shape)
		print("y: ", y.shape)
		X = np.concatenate((X, x), axis=0)
		Y = np.concatenate((Y, y), axis=0)
		n_files += 1
		if (n_files >= max_n_files): break
	shape = X.shape
	X = X[1:]
	Y = Y[1:]
	print(shape)
	return X, Y

def get_spect_data(max_n_files, time_steps):
	input_files = os.listdir("%s/%s/"%(data_root, spect_folder))
	X = []
	n_files = 0
	for input_file in input_files:
		if (not input_file.endswith(".npy")): continue
		length = len(input_file)
		music_name = input_file[:(length-5)]
		print(music_name)
		data = np.load("%s/%s/%s"%(data_root, spect_folder, input_file))
		length = data.shape[0]
		bi = 0
		while (True):
			start = bi * time_steps
			end = (bi+1) * time_steps
			if (end > length): break
			example = data[start:end]
			X.append(example)
			bi += 1

		n_files += 1
		if (n_files >= max_n_files): break
	X = np.array(X)
	print("data shape", X.shape)
	return X

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

def hybrid_loss(y_true, y_pred):
    mse = losses.mean_squared_error(y_true, y_pred)
    corr = correlation_coefficient_loss(y_true, y_pred)
    knob = 10
    loss = corr + knob * mse
    return loss

class GAN():
	def __init__(self, n_features, n_units, time_steps):
		self.n_features = n_features
		self.time_steps = time_steps # 2000? 1000?
		self.n_units = n_units

		# Build and compile the discriminator
		self.discriminator = self.build_discriminator()
		self.discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

		# Build and compile the generator
		self.generator = self.build_generator()

		seed = Input(shape=(None, self.n_features))
		gen_audio = self.generator(seed) # Here is the problem

		# For the combined model we will only train the generator
		self.discriminator.trainable = False

		# The discriminator takes generated images as input and determines validity
		valid = self.discriminator(gen_audio)

		# The combined model (stacked generator and discriminator) takes
		# noise as input => generates audio => determines validity
		self.combined = Model(seed, valid)
		self.combined.compile(loss='binary_crossentropy', optimizer='adam') 

	def predict(self, seed):
		print("Predict sequence of length %d" %self.time_steps)
		# Initialize states
		h = np.zeros((1, self.n_units))
		c = np.zeros((1, self.n_units))
		state = [h, c]
		# start of sequence input
		target_seq = seed
		# collect predictions
		output = list()
		for t in range(self.time_steps):
			# predict next stepl
			yhat, h, c = self.generator.predict([target_seq] + state)
			new = yhat[0, -1, :]
			# print("yhat shape", yhat.shape)
			# store prediction
			output.append(new) 
			# update state
			state = [h, c]
			# update target sequence
			target_seq = new.reshape((1, 1, new.shape[0]))
			# if (t % 500 == 0):
			# 	print("t=%d" %t)

			output = np.array(output)
			return output

	# def build_generator(self):
	# 	noise_shape = (None, self.n_features)

	# 	model = Sequential()
	# 	model.add(LSTM(self.n_units, input_shape=noise_shape, return_sequences=True, return_state=True))
	# 	model.add(Dense(self.n_features, activation='sigmoid'))

	# 	model.summary()

	# 	noise = Input(shape=noise_shape)

	# 	audio, _, _ = model(noise)

	# 	return Model(noise, audio)

	def build_generator(self):
		inputs = Input(shape=(None, self.n_features))
		lstm_layer = LSTM(self.n_units, return_sequences=True)
		dense_layer = Dense(self.n_features, activation='sigmoid')

		outputs = lstm_layer(inputs)
		outputs = dense_layer(outputs)
		generator = Model(inputs, outputs)

		return generator


	def build_discriminator(self):
		# An LSTM model that returns a number between 0 and 1
		audio_shape = (None, self.n_features)

		model = Sequential()
		model.add(LSTM(self.n_units, input_shape=audio_shape))
		model.add(Dense(1, activation='sigmoid'))

		model.summary()

		audio = Input(shape=audio_shape)
		validity = model(audio)

		return Model(audio, validity)

	# def build_discriminator(self):
	# 	inputs = Input(shape=(None, n_features))

	# 	# Define layers
	# 	lstm_layer = LSTM(self.n_units)
	# 	dense_layer = Dense(self.n_features, activation='sigmoid')

	# 	# Training model
	# 	outputs = lstm_layer(inputs)
	# 	outputs = dense_layer(outputs)
	# 	train_model = Model(inputs, outputs)



	def train(self, X_train, epochs, batch_size=128, sample_interval=50):
		# Load dataset
		# length = X_train.shape[0]
		# pruned_length = length - (length%self.time_steps)
		d_losses = []
		g_losses = []
		accuracies = []

		half_batch = int(batch_size / 2)

		for epoch in range(epochs):

			# -----------------------
			#  Train Discriminator 
			# -----------------------

			# Select a random half batch of audio_data
			idx = np.random.randint(0, X_train.shape[0], half_batch)
			audio = X_train[idx] # (half_batch, time_steps, n_features)


			# Choose a random piece of consecutive training data
			n_idx = np.random.randint(0, X_train.shape[0], half_batch)
			noise = X_train[n_idx] # (half_batch, time_steps, n_features)

			# Generate a half batch of new audio
			gen_audio = self.generator.predict(noise) # (half_batch, time_steps, n_features)

			d_loss_real = self.discriminator.train_on_batch(audio, np.ones((half_batch, 1)))
			d_loss_fake = self.discriminator.train_on_batch(gen_audio, np.zeros((half_batch, 1)))
			d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

			# -----------------------
			#  Train Generator
			# -----------------------

			n_idx = np.random.randint(0, X_train.shape[0], batch_size)
			noise = X_train[n_idx] # (half_batch, time_steps, n_features)

			# The generator wants the discriminator to label the generated samples
			# as valid (ones)
			valid_y = np.array([1] * batch_size) 

			# Train the generator
			g_loss = self.combined.train_on_batch(noise, valid_y)

			# Plot the progress
			# print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
			print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
			d_losses.append(d_loss[0])
			g_losses.append(g_loss)
			accuracies.append(d_loss[1])

			# If at save interval => save generated audio samples
			if (epoch % sample_interval == 0):
				noise = random.choice(X_train)
				noise = noise.reshape(1, noise.shape[0], noise.shape[1])
				self.sample_audio(epoch, noise)

		return (d_losses, g_losses, accuracies)

	def sample_audio(self, epoch, noise):
		gen_audio = self.generator.predict(noise) # gen_audio is a spectrogram (1, time_steps, n_features)
		gen_audio = gen_audio.reshape(gen_audio.shape[1], gen_audio.shape[2])
		#print(gen_audio[1000:1010, :5])

		new_reconstruct_ifft.convert_back_wav("results/gan_10songs_%d.wav"%epoch, gen_audio)

def plot_metric(d_losses, g_losses):
	plt.plot(d_losses)
	plt.plot(g_losses)
	plt.title("GAN loss")
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Discriminator', 'Generator'], loc='upper right')
	plt.savefig('plot/gan_10songs.png')
	plt.close()

def plot_accuracy(accuracies):
	plt.plot(accuracies)
	plt.title("Discriminator Accuracy")
	plt.ylabel('Accuracy')
	plt.xlabel('epoch')
	plt.savefig('plots/acc_10songs.png')
	plt.close()

if __name__ == '__main__':
	n_features = 40
	n_units = 256
	time_steps = 2000 # Change this later
	gan = GAN(n_features, n_units, time_steps)
	X = get_spect_data(10, time_steps)
	(d_losses, g_losses, accuracies) = gan.train(X, epochs=100, batch_size=32, sample_interval=5)
	plot_metric(d_losses, g_losses)
	plot_accuracy(accuracies)



