import numpy as np
import scipy.io.wavfile as wav

def reconstruct(window):
	result = []
	if len(window) == 0: return result
	block_len = window.shape[1]/2
	window[:][:] *= 100
	window[:][:] -= 50
	for block in window:
		real_part = block[0:block_len]
		imag_part = block[block_len:]
		conv_block = real_part + 1.0j * imag_part
		res = np.fft.ifft(conv_block)
		result.append(res)
	result = np.array(result)
	result = np.real(result.flatten())
	return result

# output to filename: e.g. "out.wav"
def convert_back_wav(filename, window):
	res_data = reconstruct(window)
	back_song = res_data * 32767.0
	song_new = back_song.astype('int16')
	wav.write(filename, 44100, song_new)
