import numpy as np
from __future__ import print_function, division

class GAN():
	def __init__(self, discriminator, generator, dataset):
		self.discriminator = discriminator
		self.generator = generator
		self.data = dataset
		self.sample_size = dataset.shape[0]
		self.time_step = dataset.shape[1]
		self.block_size = dataset.shape[2]

	def train(self, epochs):
		for epoch in range(epochs):
			# -------------------
			# Train Discriminator
			# -------------------

			# Generate new audio
			seed = np.random.rand(1, 1, self.block_size)
			gen_audio = self.generator.predict(seed)

			# Train the discriminator
			self.discriminator.train(self.data, np.ones(self.sample_size))
			self.discriminator.train(gen_audio, np.one(1))

			# -------------------
			# Train Generator
			# -------------------

			seed = np.random.rand(1, 1, self.block_size)
			valid_y = np.array([1] * self.sample_size)
			self.combined.train(noise,valid_y)


