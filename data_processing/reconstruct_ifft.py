import numpy as np

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
	return result

