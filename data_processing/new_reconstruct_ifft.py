from __future__ import division
import numpy as np
import scipy
import scipy.io.wavfile as siow

def reconstruct(window):
	result = []
	if len(window) == 0: return result
	block_len = int(window.shape[1]/2)
	# window[:][:] *= 100
	# window[:][:] -= 50
	for block in window:
		real_part = block[0:block_len]
		imag_part = block[block_len:]
		conv_block = real_part + 1.0j * imag_part
		res = np.fft.ifft(conv_block)
		result.append(res)
	result = np.array(result)
	result = np.real(result.flatten())
	return result

def convert_back_wav(filename, window):
	rate = 44100
	data = reconstruct(window)
	data *= 32767.0
	data = data.astype('int16')
	# res_data = res_data.astype('int16')
	length = data.shape[0]
	half_length = int(length/2)
	print("generated data", data[half_length:(half_length+100)])
	siow.write(filename, rate, data)
