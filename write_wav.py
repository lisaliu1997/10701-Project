import scipy
import scipy.io.wavfile as siow

def write2wav(data, rate, filename):
	print(type(data))
	siow.write(filename, rate, data)
