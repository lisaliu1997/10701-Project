"""
Convert mp3 files to wav formatted files
"""
from pipes import quote
import os

# takes in a mp3 and convert to wav using lame
def mp3_to_wav_file(mp3_file, mp3_folder, wav_folder, tmp_mp3_folder):
	full_mp3_file = mp3_folder + '/' + mp3_file
	tmp_mp3_file = tmp_mp3_folder + '/' + mp3_file
	wav_file_name = wav_folder + '/' + mp3_file[:-4] + '.wav'
	sample_freq = "44.1"
	# produce mono .mp3 file
	cmd_mp3 = 'lame -m m -a {0} {1}'.format(quote(full_mp3_file), quote(tmp_mp3_file))
	os.system(cmd_mp3)
	# decode to wav file
	cmd_wav = 'lame --decode {0} {1} --resample {2}'.format(quote(tmp_mp3_file), quote(wav_file_name), sample_freq)
	os.system(cmd_wav)
	return

# takes in a folder of mp3 files and convert to wav file
def all_mp3_to_wav(mp3_folder):
	tmp_mp3_folder = mp3_folder + '/' + 'tmp_mp3'
	wav_folder = mp3_folder + '/' + 'wav'
	if not os.path.exists(tmp_mp3_folder):
		os.makedirs(tmp_mp3_folder)
	if not os.path.exists(wav_folder):
		os.makedirs(wav_folder)
	for file in os.listdir(mp3_folder):
		if (file.endswith('.mp3')):
			mp3_to_wav_file(file, mp3_folder, wav_folder, tmp_mp3_folder)
	return

