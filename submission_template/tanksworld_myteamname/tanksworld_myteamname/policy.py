import numpy as np
import random

class Policy:

	def __init__(self):
		""" Do any initial setup here """
		pass

	def game_reset(self):
		""" Do any game start setup here """
		pass

	def get_actions(self, states):
		
		#double checking some data sizes
		assert len(states) == 5
		for s in states:
			assert s.shape == (128,128,4)

		#return 5 random actions, each of length 3, in range [-1,1]
		return [[random.uniform(-1,1) for _ in range(3)] for _ in range(5)]