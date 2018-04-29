import numpy as np
import write_wav

# takes in a 2D array and reconstruct to 1D by applying ifft 
def reconstruct(window):
	result = []
	if len(window) == 0: return result
	block_len = window.shape[1]/2
	window[:][:] *= 500000
	for block in window:
		real_part = block[0:block_len]
		imag_part = block[block_len:]
		conv_block = real_part + 1.0j * imag_part
		res = np.fft.ifft(conv_block)
		result.append(res)
	result = np.array(result)
	result = np.real(result.flatten())
	return result

conv_arr = np.load('x_dir/a_x.npy')
(a, b, c) = conv_arr.shape
new_conv_arr = conv_arr.reshape(a*b, c)
rate = 10000
audio_data = reconstruct(new_conv_arr)
siow.write(audio_data, 1000, data)
write_wav.write2wav(new_conv_arr, rate, "results/1.wav")

