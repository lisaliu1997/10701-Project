import lstm_model
import write_wav
import numpy as np
import random


time_steps = 20
n_features = 100
n_units = 256

def get_data():
	X = np.load("data/music_out_x.npy")
	Y = np.load("data/music_out_y.npy") 
	return X, Y

def main():
	X, Y = get_data()
	train_model = lstm_model.build_train(X, Y, n_features, n_units)
	rate = 10000
	seconds = 0.5
	n_steps = int(seconds * rate)
	seed = random.choice(X)
	seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	output = lstm_model.predict_sequence(train_model, seed, n_steps, n_features)
	write_wav.write2wav(output, rate, "results/1.wav")

if __name__ == "__main__":
	main()
