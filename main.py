import model
import write_wav

def main():
	X, Y = get_data()
	train_model = model.build_train(X, Y, n_features, n_units)
	rate = 10000
	seconds = 50
	n_steps = seconds * rate
	output = model.predict_sequence(model, n_steps, n_features)
	write_wave.write2wav(output, rate, "results.wav")
