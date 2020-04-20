import sys, time, random
import gym
from mlagents.envs import UnityEnvironment
import os
from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from tanksworld.minimap_util import minimap_for_player, displayable_rgb_map, display_cvimage
import cv2
import pathlib
#from .utils import get_l2explorer_worker_id, get_l2explorer_app_location

# Enforce Python 3.6.x (the only version supported by Unity MLAgents)
if not (sys.version_info >= (3, 6, 0) and sys.version_info < (3, 7, 0)):
    raise RuntimeError('Python 3.6 required. Current version is ' + sys.version)


class TanksWorldEnv(gym.Env):
    #Singleton Implementation of unity environment
    _env = None
    _env_params = {}
    _MAX_INT = 2147483647 #Max int for Unity ML Seed
    _tank_data_len = 9

    @classmethod
    def close_env(cls):
        """Close the Unity environment and reset all environment variables"""
        if TanksWorldEnv._env:
            TanksWorldEnv._env.close()
        TanksWorldEnv._env = None
        TanksWorldEnv._env_params = []

    #DO this in reset to allow seed to be set
    def __init__(self, exe, action_repeat=6, image_scale=128, timeout=500, friendly_fire=True, take_damage_penalty=True, kill_bonus=True, death_penalty=True,
        static_tanks=[], random_tanks=[], disable_shooting=[], penalty_weight=1.0, reward_weight=1.0, will_render=False):

        # call reset() to begin playing
        self._workerid = MPI.COMM_WORLD.Get_rank() #int(os.environ['L2EXPLORER_WORKER_ID'])
        self._filename =  exe#'/home/rivercg1/projects/aisafety/build/aisafetytanks_0.1.2/TanksWorld.x86_64'
        self.observation_space = None
        self.observation_space = gym.spaces.box.Box(0,255,(128,128,4))
        self.action_space = gym.spaces.box.Box(-1,1,(3,))
        self.action_space = None
        self._seed = None

        self.timeout = timeout
        self.action_repeat=action_repeat  # repeat action this many times
        self.image_scale = image_scale    # scale the images going to the tanks (will be scaled from 128)
        self.penalties = friendly_fire    # include negative rewards for friendly fire, neutral tanks, etc
        self.take_damage_penalty = take_damage_penalty #penalize getting hit by someone else
        self.kill_bonus = kill_bonus      # get bonus +1 for killing enemy (and -1 for ally if friendly fire on)
        self.death_penalty = death_penalty # get penalty of -1.0 when killed

        self.reward_weight = reward_weight
        self.penalty_weight = penalty_weight

        self.static_tanks = static_tanks
        self.random_tanks = random_tanks
        self.disable_shooting = disable_shooting
        self.will_render = will_render

        for s in static_tanks:
            assert s not in random_tanks

        self.training_tanks = []
        for i in range(10):
            if i not in self.static_tanks and i not in self.random_tanks:
                self.training_tanks.append(i)

        #load the obstaces image
        #path = pathlib.Path(module.__file__).resolve().parent
        self.barrier_img = cv2.imread('./obstaclemap_fixed.png',1)

        self.reset(params={})

    def seed(self, val):
        self._seed = int(val)%TanksWorldEnv._MAX_INT #integer seed required, convert

    def get_state(self):
        state = self._env_info.vector_observations[0]
        state_reformat = []
        for i in range(12):
            j = i*TanksWorldEnv._tank_data_len
            refmt = [state[j+0],state[j+1],state[j+2]/180*3.1415,state[j+3], state[j+7], state[j+8]]

            state_reformat.append(refmt)

        barriers = self.barrier_img/255.0  #np.array(self._env_info.visual_observations[1][0])

        ret_states = [minimap_for_player(state_reformat,i,barriers) for i in self.training_tanks]

        if self.will_render:
            self.disp_states = [displayable_rgb_map(s) for s in ret_states]

        if self.image_scale != 128:
            ret_states = [cv2.resize(s, (self.image_scale, self.image_scale)) for s in ret_states]

        return ret_states

    def reset(self,**kwargs):
        # Reset the environment
        params ={}
        self.dead = []
        if 'params' in kwargs:
            params = kwargs['params']
        if not TanksWorldEnv._env:
            try:
                print('WARNING: seed not set, using default')
                TanksWorldEnv._env = UnityEnvironment(file_name=self._filename, worker_id=self._workerid, seed=1234,timeout_wait=500)
                print('finished initializing environment')
                TanksWorldEnv._env_params['filename'] = self._filename
                TanksWorldEnv._env_params['workerid'] = self._workerid
            except:
                print('ERROR: could not initialize unity environment, are filename correct and workerid not already in use by another unity instance?')
                raise

        # Set the default brain to work with
        self._default_brain = self._env.brain_names[0]
        print("number of brains ", len(self._env.brain_names))
        brain = self._env.brains[self._default_brain]
        self._env_info = self._env.reset(train_mode=0, config=params)[self._default_brain]

        self.previous_health = [100.0]*12

        self.episode_steps = 0

        state = self.get_state()

        return state

    def is_done(self,state):
        red_health = [i*TanksWorldEnv._tank_data_len + 3 for i in range(5)]
        blue_health = [(i+5)*TanksWorldEnv._tank_data_len + 3 for i in range(5)]
        training_health = [i*TanksWorldEnv._tank_data_len + 3 for i in self.training_tanks]

        red_dead = [state[i]<=0 for i in red_health]
        blue_dead = [state[i]<=0 for i in blue_health]
        training_dead = [state[i]<=0 for i in training_health]

        if  all(red_dead) or all(blue_dead) or all(training_dead) or self.episode_steps>self.timeout:
            return True
        return False


    def objectives(self):
        health = [self._env_info.vector_observations[0][i*TanksWorldEnv._tank_data_len + 3] for i in range(12)]
        delta_health = [self.previous_health[i]-health[i] for i in range(12)]

        reward = [0.0]*10

        for i in range(10):
            state = self._env_info.vector_observations[0][i*TanksWorldEnv._tank_data_len:(i+1)*TanksWorldEnv._tank_data_len]
            damage_dealt = state[5]/100.0
            if state[4] == 1 and self.kill_bonus:
                damage_dealt += 1.0
            my_team = 1 if i<5 else 2
            team_hit = state[6]

            # if team_hit > 0:  [TED] I don't think this should have been here- what if we hit neutral?

            # was the damage dealth good or bad?
            if (my_team==1 and team_hit==2) or (my_team==2 and team_hit==1):
                multiplier = self.reward_weight
            else:
                multiplier = -self.penalty_weight

            #eliminate friendly fire (and neutral) penalties if required
            if multiplier < 0 and not self.penalties:
                multiplier = 0.0

            reward[i] += multiplier * damage_dealt

            if delta_health[i] != 0.0:
                if health[i] <= 0.0 and self.death_penalty:
                    reward[i] -= 1.0 * self.penalty_weight
                elif self.take_damage_penalty:
                    reward[i] -= delta_health[i] * self.penalty_weight / 100.0

        self.previous_health = health
        return [reward[i] for i in self.training_tanks]


    def step(self, action):

        action = action[:]

        self.reward = [0.0]*len(self.training_tanks)

        random_actions = []
        for i in self.random_tanks:
            random_actions.append([random.uniform(-1.0,1.0), random.uniform(-1.0,1.0), random.uniform(-1.0,1.0)])

        for stp in range(self.action_repeat):
            self.episode_steps += 1

            #initialize to all static
            new_action = []
            for i in range(10):
                new_action.append([0.0,0.0,0.0])

            #sub in real actions for training tanks
            for trainidx, totalidx in enumerate(self.training_tanks):
                new_action[totalidx] = action[trainidx]

            #sub in random actions for random tanks
            for randidx, totalidx in enumerate(self.random_tanks):
                new_action[totalidx] = random_actions[randidx]

            #turn off shooting for any tanks with disabled shooting
            for didx, totalidx in enumerate(self.disable_shooting):
                new_action[totalidx][-1] = -1.0

            #step
            new_action = np.array(new_action)
            self._env_info = self._env.step(new_action)[self._default_brain]

            #get state
            self.state = self.get_state()

            #add rewards
            new_rew = self.objectives()
            for i in range(len(new_rew)):
                self.reward[i] += new_rew[i]

            #done
            self.done = self._env_info.local_done[0]

            if self.done:
                break

        info = [{}]*len(self.training_tanks)
        return self.state, self.reward, self.done or self.is_done(self._env_info.vector_observations[0]), info

    def render(self):
        if self.will_render:
            for idx,s in enumerate(self.disp_states):
                display_cvimage("player_"+str(idx), s)




class TanksWorldStackedEnv(TanksWorldEnv):
    #Singleton Implementation of unity environment
    _env = None
    _env_params = {}
    _MAX_INT = 2147483647 #Max int for Unity ML Seed


    #DO this in reset to allow seed to be set
    def __init__(self,exe):
        # call reset() to begin playing
        self._workerid = MPI.COMM_WORLD.Get_rank() #int(os.environ['L2EXPLORER_WORKER_ID'])
        self._filename =  exe#'/home/rivercg1/projects/aisafety/build/aisafetytanks_0.1.2/TanksWorld.x86_64'
        #self.observation_space = None
        self.stack = 3
        self.observation_space = gym.spaces.box.Box(0,255,(128,128,4*self.stack))
        self.action_space = gym.spaces.box.Box(-1,1,(3,))
        self.action_space = None
        self._seed = None

        self.reset(params={})

    def reset(self,**kwargs):
        # Reset the environment
        params ={}
        self.dead = []
        self.states = []
        if 'params' in kwargs:
            params = kwargs['params']
        if not TanksWorldStackedEnv._env:
            try:
                print('WARNING: seed not set, using default')
                TanksWorldStackedEnv._env = UnityEnvironment(file_name=self._filename, worker_id=self._workerid, seed=1234,timeout_wait=500)
                print('finished initializing environment')
                TanksWorldStackedEnv._env_params['filename'] = self._filename
                TanksWorldStackedEnv._env_params['workerid'] = self._workerid
            except:
                print('ERROR: could not initialize unity environment, are filename correct and workerid not already in use by another unity instance?')
                raise

        # Set the default brain to work with
        self._default_brain = self._env.brain_names[0]
        print("number of brains ", len(self._env.brain_names))
        brain = self._env.brains[self._default_brain]
        self._env_info = self._env.reset(train_mode=0, config=params)[self._default_brain]


        state = self.get_state()

        return state

    def get_state(self):
        state = self._env_info.vector_observations[0]
        state_reformat = [[state[i+0],state[i+1],state[i+2]/180*3.1415,state[i+3]] for i in range(0,84,7)]
        barriers = np.array(self._env_info.visual_observations[1][0])
        #state_modified = [minimap_for_player(state_reformat,i,barriers) for i in range(10)]

        for i in range(10):
            state = minimap_for_player(state_reformat,i,barriers)
            #print("state shape ",np.array(state).shape)
            if len(self.states)<10:
                self.states.append([state]*self.stack)
            else:
                #print("step 1", len(self.states[i]), np.array(self.states[i][0]).shape)
                self.states[i].pop(0)
                #print("step 2", len(self.states[i]), np.array(self.states[i][0]).shape)
                self.states[i].append(state)
                #print("step 3", len(self.states[i]), np.array(self.states[i][0]).shape)

        #print('result shape ', np.array(self.states[0]).shape)
        result = [np.array(self.states[i]).squeeze().transpose((1,2,0,3)).reshape((128,128,-1)) for i in range(10)]
        #print('result length ' ,len(result), result[i].shape)
        return result
