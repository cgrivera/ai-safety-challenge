#(c) 2019 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).
#All Rights Reserved. This material may be only be used, modified, or reproduced
#by or for the U.S. Government pursuant to the license rights granted under the
#clauses at DFARS 252.227-7013/7014 or FAR 52.227-14. For any other permission,
#please contact the Office of Technology Transfer at JHU/APL.
#
#NO WARRANTY, NO LIABILITY. THIS MATERIAL IS PROVIDED "AS IS." JHU/APL MAKES NO
#REPRESENTATION OR WARRANTY WITH RESPECT TO THE PERFORMANCE OF THE MATERIALS,
#INCLUDING THEIR SAFETY, EFFECTIVENESS, OR COMMERCIAL VIABILITY, AND DISCLAIMS
#ALL WARRANTIES IN THE MATERIAL, WHETHER EXPRESS OR IMPLIED, INCLUDING (BUT NOT
#LIMITED TO) ANY AND ALL IMPLIED WARRANTIES OF PERFORMANCE, MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT OF INTELLECTUAL PROPERTY
#OR OTHER THIRD PARTY RIGHTS. ANY USER OF THE MATERIAL ASSUMES THE ENTIRE RISK
#AND LIABILITY FOR USING THE MATERIAL. IN NO EVENT SHALL JHU/APL BE LIABLE TO ANY
#USER OF THE MATERIAL FOR ANY ACTUAL, INDIRECT, CONSEQUENTIAL, SPECIAL OR OTHER
#DAMAGES ARISING FROM THE USE OF, OR INABILITY TO USE, THE MATERIAL, INCLUDING,
#BUT NOT LIMITED TO, ANY DAMAGES FOR LOST PROFITS.

import sys
import gym
from mlagents.envs import UnityEnvironment
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
#from .utils import get_l2explorer_worker_id, get_l2explorer_app_location

# Enforce Python 3.6.x (the only version supported by Unity MLAgents)
if not (sys.version_info >= (3, 6, 0) and sys.version_info < (3, 7, 0)):
    raise RuntimeError('Python 3.6 required. Current version is ' + sys.version)


class TanksWorldFPVEnv(gym.Env):
    #Singleton Implementation of unity environment
    _env = None
    _env_params = {}
    _MAX_INT = 2147483647 #Max int for Unity ML Seed

    @classmethod
    def close_env(cls):
        """Close the Unity environment and reset all environment variables"""
        if TanksWorldFPVEnv._env:
            TanksWorldFPVEnv._env.close()
        TanksWorldFPVEnv._env = None
        TanksWorldFPVEnv._env_params = []

    #DO this in reset to allow seed to be set
    def __init__(self):
        # call reset() to begin playing
        self._workerid = MPI.COMM_WORLD.Get_rank() #int(os.environ['L2EXPLORER_WORKER_ID'])
        #self._filename='/home/rivercg1/projects/l2m/exe/machine/l2explorer_0_1_0_linux_machine.x86_64'
        self._filename =  '/home/rivercg1/projects/aisafety/build/aisafetytanks_0.1.0/tanks.x86_64'
        self.observation_space = None
        self.action_space = None
        self._seed = None
        self.reset(params={})

    def seed(self, val):
        self._seed = int(val)%TanksWorldFPVEnv._MAX_INT #integer seed required, convert

    def reset(self,**kwargs):
        # Reset the environment
        params ={}
        if 'params' in kwargs:
            params = kwargs['params']
        if not TanksWorldFPVEnv._env:
            try:
                print('WARNING: seed not set, using default')
                TanksWorldFPVEnv._env = UnityEnvironment(file_name=self._filename, worker_id=self._workerid, seed=1234)
                print('finished initializing environment')
                TanksWorldFPVEnv._env_params['filename'] = self._filename
                TanksWorldFPVEnv._env_params['workerid'] = self._workerid
            except:
                print('ERROR: could not initialize unity environment, are filename correct and workerid not already in use by another unity instance?')
                raise

        # Set the default brain to work with
        self._default_brain = self._env.brain_names[0]
        print("number of brains ", len(self._env.brain_names))
        brain = self._env.brains[self._default_brain]
        #lets pretend that i'm tank at index 3 , this will have to be fixed later
        self._env_info = self._env.reset(train_mode=0, config=params)[self._default_brain]
        self.observation_space = gym.spaces.box.Box(0,255,(84,84,3))
        self.action_space = gym.spaces.box.Box(-1,1,(3,))

        state = self._env_info.visual_observations[0]

        return state

    def step(self, action):
        #render and get state as an image
        action = np.array(action)
        self._env_info = self._env.step(action)[self._default_brain]

        self.state = self._env_info.visual_observations[0]
        self.reward  = self._env_info.rewards
        self.done = self._env_info.local_done
        info = [{}]*10

        print('stepping ',self.reward,self.done)
        print('lengths ',len(self.state),len(self.reward),len(self.done),len(info))
        return self.state, self.reward, self.done[0], info

    def render(self):
        '''Render by returning visual observation 1 for display'''
        return self._env_info.visual_observations[0]
