import lstm_model as model
import write_wav
import numpy as np
import random
import os

time_steps = 20
n_features = 40
n_units = 256

data_root = "data"
input_folder = "x_dir"
output_folder = "y_dir"

def get_data(max_n_files):
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

def main():
	X, Y = get_data(10)

	# build and train model
	inf_model = model.build_train(X, Y, n_features, n_units)

	# Make predictions and generate music
	rate = 10000
	seconds = 0.4
	n_steps = int(seconds * rate)
	seed = random.choice(X)
	n_examples = X.shape[0]
	files = os.listdir("%s/%s/"%(data_root, input_folder))
	files = list(filter(lambda x: x.endswith("x.npy"), files))
	test_file = random.choice(files)
	test_x = np.load("%s/%s/%s"%(data_root, input_folder, test_file))
	test_x = np.reshape(test_x, (1, test_x.shape[0]*test_x.shape[1], test_x.shape[2]))
	seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	model.predict_sequence(inf_model, test_x, n_steps, n_features, n_units)
	# write_wav.write2wav(output, rate, "results/2-41.wav")

if __name__ == "__main__":
	main()

