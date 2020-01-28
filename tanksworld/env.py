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
from minimap_util import minimap_for_player
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
    def __init__(self,exe):
        # call reset() to begin playing
        self._workerid = MPI.COMM_WORLD.Get_rank() #int(os.environ['L2EXPLORER_WORKER_ID'])
        #self._filename='/home/rivercg1/projects/l2m/exe/machine/l2explorer_0_1_0_linux_machine.x86_64'
        self._filename =  exe#'/home/rivercg1/projects/aisafety/build/aisafetytanks_0.1.2/TanksWorld.x86_64'
        self.observation_space = None
        self.action_space = None
        self._seed = None
        self.reset(params={})

    def seed(self, val):
        self._seed = int(val)%TanksWorldFPVEnv._MAX_INT #integer seed required, convert

    def get_state(self):
        state = self._env_info.vector_observations[0]
        #for i in range(0,84,7):
        print('image shape ', np.array(self._env_info.visual_observations[1][0]).shape)
        #plt.imshow(np.array(self._env_info.visual_observations[1][0]))
        state_reformat = [[state[i+0],state[i+1],state[i+2]/180*3.1415,state[i+3]] for i in range(0,84,7)]
        barriers = np.array(self._env_info.visual_observations[1][0])
        #print('state shape ',np.squeeze(np.array(minimap_for_player(state_reformat,3)).transpose((3,1,2,0))).shape)
        state_modified = [np.squeeze(np.array(minimap_for_player(state_reformat,i,barriers)).transpose((3,1,2,0))) for i in range(10)]
        return state_modified

    def reset(self,**kwargs):
        # Reset the environment
        params ={}
        self.dead = []
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
        self.observation_space = gym.spaces.box.Box(0,255,(128,128,4))
        self.action_space = gym.spaces.box.Box(-1,1,(3,))

        state = self.get_state()

        return state

    def is_done(self,state):
        red_health = [3,10,17,24,31]
        blue_health = [38,45,52,59,66]

        red_dead = [state[i]<=0 for i in red_health]
        blue_dead = [state[i]<=0 for i in blue_health]
        #print('red ',[state[i] for i in red_health], ' blue_dead ',[state[i] for i in blue_health])

        if  all(red_dead) or all(blue_dead):
            return True
        return False

    def objective(self,num):
        state = self._env_info.vector_observations[0]
        index = num*7
        if state[index+3]<=0 and num not in self.dead:
            self.dead.append(num)
            return -1.0
        if state[index+4]==1.0:
            print("tank no ", num,' state vector ', [state[index+i] for i in range(7)])
        if state[index+4]==1.0 and ((num<5 and state[index+6]==2.0) or (num>=5 and num<10 and state[index+5]==1.0)):
            print("******************Got a hit!***********************")
            print("tank no ", num,' state vector ', [state[index+i] for i in range(7)])
            return .125
        return 0.0

    def step(self, action):
        #render and get state as an image
        action = np.array(action)
        self._env_info = self._env.step(action)[self._default_brain]

        self.state = self.get_state()
        #self.reward  = self._env_info.rewards
        self.reward = [self.objective(i) for i in range(10)]
        self.done = self._env_info.local_done
        info = [{}]*10

        #print('stepping ',self.reward,self.done)
        #print('lengths ',len(self.state),len(self.reward),len(self.done),len(info))
        #print('visual observation shapes ',self._env_info.vector_observations[0])
        print('reward ',self.reward)
        return self.state, self.reward, self.is_done(self._env_info.vector_observations[0]), info

    def render(self):
        '''Render by returning visual observation 1 for display'''
        return self._env_info.visual_observations[0]
