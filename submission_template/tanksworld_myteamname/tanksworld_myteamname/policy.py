import numpy as np
import random

class Policy:

	def __init__(self):
		""" Do any initial setup here """
		pass

	def game_reset(self, tank_id):
		""" Do any game start setup here """
		pass

	def get_actions(self, state, info=None):
		
		#double checking some data sizes
		assert state.shape == (128,128,4)

		#return random actions
		return [random.uniform(-1,1) for _ in range(3)]
