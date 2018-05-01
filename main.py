import lstm_model as model
import write_wav
import numpy as np
import random
import os

time_steps = 40
n_features = 22050
n_units = 1024

# data_root = "data"
# input_folder = "x_dir"
# output_folder = "y_dir"
# spect_folder = 'spectrogram'

# def get_data(max_n_files):
# 	input_files = os.listdir("%s/%s/"%(data_root, input_folder))
# 	X = np.zeros((1, time_steps, n_features))
# 	Y = np.zeros((1, time_steps, n_features))
# 	n_files = 0
# 	for input_file in input_files:
# 		if (not input_file.endswith("x.npy")): continue
# 		length = len(input_file)
# 		music_name = input_file[:(length-5)]
# 		print(music_name)
# 		x = np.load("%s/%s/%s"%(data_root, input_folder, input_file))
# 		output_file = "%sy.npy"%music_name
# 		output_file_path = "%s/%s/%s"%(data_root, output_folder, output_file)
# 		if (not os.path.isfile(output_file_path)): continue
# 		y = np.load(output_file_path)
# 		print("x: ", x.shape)
# 		print("y: ", y.shape)
# 		X = np.concatenate((X, x), axis=0)
# 		Y = np.concatenate((Y, y), axis=0)
# 		n_files += 1
# 		if (n_files >= max_n_files): break
# 	shape = X.shape
# 	X = X[1:]
# 	Y = Y[1:]
# 	print(shape)
# 	return X, Y

# def get_spect_data(max_n_files):
# 	input_files = os.listdir("%s/%s/"%(data_root, spect_folder))
# 	X, Y = [], []
# 	n_files = 0
# 	for input_file in input_files:
# 		if (not input_file.endswith("x.npy")): continue
# 		length = len(input_file)
# 		music_name = input_file[:(length-5)]
# 		print(music_name)
# 		data = np.load("%s/%s/%s"%(data_root, spect_folder, input_file))
# 		length = data.shape[0]
# 		bi = 0
# 		while (True):
# 			start = bi * time_steps
# 			end = (bi+1) * time_steps
# 			if (end > length): break
# 			example = data[start:end]
# 			y = example
# 			x = np.concatenate((np.array([[0. for _ in range(n_features)]]), example[:-1]))
# 			x = np.flip(x, 0).copy()
# 			X.append(x)
# 			Y.append(y)
# 			bi += 1

# 		n_files += 1
# 		if (n_files >= max_n_files): break
# 	X, Y = np.array(X), np.array(Y)
# 	print("data shape", X.shape, Y.shape)
# 	return X, Y

def get_data_frm_file():
	root = '5_piano'
	input_file = 'YourMusicLibraryNP_x.npy'
	output_file = 'YourMusicLibraryNP_y.npy'
	X = np.load("%s/%s"%(root, input_file))
	Y = np.load("%s/%s"%(root, output_file))
	return (X, Y)

# Copies a random example's first seed_length sequences as input to the generation algorithm
def generate_copy_seed_sequence(seed_length, training_data):
	num_examples = training_data.shape[0]
	example_len = training_data.shape[1]
	randIdx = np.random.randint(num_examples, size=1)[0]
	randSeed = np.concatenate(tuple([training_data[randIdx + i] for i in range(seed_length)]), axis=0)
	seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))
	return seedSeq

def main():
	# X, Y = get_data(1)
	(X, Y) = get_data_frm_file()

	# build and train model
	inf_model = model.build_train(X, Y, n_features, n_units)

	# Make predictions and generate music
	rate = 44100
	seconds = 0.4
	n_steps = int(seconds * rate)
	seed = random.choice(X)
	print("seed")
	print(seed[:, :5])
	seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
	model.predict_sequence(inf_model, seed, n_steps, n_features, n_units)

if __name__ == "__main__":
	main()

# def main():
# 	# X, Y = get_data(1)
# 	X, Y = get_data_frm_file()

# 	# build and train model
# 	inf_model = model.build_train(X, Y, n_features, n_units)

# 	# Make predictions and generate music
# 	rate = 44100
# 	seconds = 0.4
# 	n_steps = int(seconds * rate)
# 	seed = random.choice(X)

# 	files = os.listdir("%s/%s/"%(data_root, input_folder))
# 	files = list(filter(lambda x: x.endswith("x.npy"), files))
# 	test_file = random.choice(files)
# 	test_x = np.load("%s/%s/%s"%(data_root, input_folder, test_file))
# 	test_x = np.reshape(test_x, (1, test_x.shape[0]*test_x.shape[1], test_x.shape[2]))
# 	seed = np.reshape(seed, (1, seed.shape[0], seed.shape[1]))
# 	model.predict_sequence(inf_model, seed, n_steps, n_features, n_units)
# 	# write_wav.write2wav(output, rate, "results/2-41.wav")


