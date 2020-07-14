import numpy as np
import random
from stable_baselines.ppo1 import PPO1
import gym
#from arena5.algos.ppo.ppo1_mod import PPO1
#import tanksworld_hivemind as module
from stable_baselines.common.policies import CnnPolicy
import os
import pathlib

class Env:
	def __init__(self):
		self.observation_space = gym.spaces.box.Box(0,255,(128,128,4))
		self.action_space = gym.spaces.box.Box(-1,1,(3,))

class Policy:

	def __init__(self):
		""" Do any initial setup here """

#		self.model = PPO1(CnnPolicy, Env(), timesteps_per_actorbatch=128, clip_param=0.2, entcoeff=0.01,optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=64, gamma=0.99, lam=0.95, schedule='linear',verbose=1)
		path = pathlib.Path(__file__).resolve().parent
		print("Loading ...",str(path)+'/ppo_save')
		self.model = PPO1.load(str(path)+'/ppo_save')

	def game_reset(self,tank_id):
		""" Do any game start setup here """
		pass

        #the info dict gets passed as well, you can make use of it or not
        #details about the reward compoents are included
        #info['blue_stats']["penalty_weight"] , in the range [0-1]
        #info['blue_stats']["friendly_fire"], [True or False]
	def get_actions(self, state,info=None):
		action,_ = self.model.predict(state)
		return action
