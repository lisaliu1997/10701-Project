import scipy
import scipy.io.wavfile as siow

(rate, data) = siow.read(filename)
siow.write(filename, rate, data)