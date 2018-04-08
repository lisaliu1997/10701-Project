import scipy
import scipy.io.wavfile as siow

(rate, data) = siow.read(filename)
# data could be a 1D array or 2D array, depending on the number of tracks.
# For simplicity, we always keep just one track.


# Use a sliding window of size 1000, step of size 100.
# Normalization: squeeze all data points into [0, 1].
# Want to get a normalized data array X of shape (num_examples, time_steps, n_features), 
# where num_examples=?, time_steps=20, n_features=50.
siow.write(filename, rate, data)