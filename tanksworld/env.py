# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
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
from tensorboardX import SummaryWriter
#from .utils import get_l2explorer_worker_id, get_l2explorer_app_location

# Enforce Python 3.6.x (the only version supported by Unity MLAgents)
if not (sys.version_info >= (3, 6, 0) and sys.version_info < (3, 7, 0)):
    raise RuntimeError('Python 3.6 required. Current version is ' + sys.version)


def team_stats_dict(env):

    return {

        #episode timing
        "frames":0,
        "frames_remaining":env.timeout,
        "episode_timeout":env.timeout,
        "action_repeat":env.action_repeat,

        #status of each team
        "tanks_alive":{"ally":5, "enemy":5, "neutral":2},
        "tanks_dead":{"ally":0, "enemy":0, "neutral":0},
        "team_health":{"ally":500.0, "enemy":500.0, "neutral":200.0},

        #firing and accuracy stats
        "number_shots_fired":{"ally":0, "enemy":0, "neutral":0},
        "number_shots_connected":{"ally":0, "enemy":0, "neutral":0},
        "target_type":{"ally":0, "enemy":0, "neutral":0},

        #damage dealt/received
        "damage_inflicted_on":{"ally":0.0, "enemy":0.0, "neutral":0.0},
        "damage_taken_by":{"ally":0.0, "enemy":0.0, "neutral":0.0},

        #tank deaths caused / experienced
        "kills_executed_on":{"ally":0, "enemy":0, "neutral":0},
        "deaths_caused_by":{"ally":0, "enemy":0, "neutral":0},

        #reward parameters given to the environment- these are constant
        "reward_parameters":{
            "reward_weight":env.reward_weight,
            "penalty_weight":env.penalty_weight,
            "friendly_fire":env.penalties,
            "take_damage_penalty":env.take_damage_penalty,
            "kill_bonus":env.kill_bonus,
            "death_penalty":env.death_penalty
        },

        #cumulative rewards by this team, before weighting and filtering
        "reward_components_cumulative":{
            "all":0.0,
            "rewards_only":0.0,
            "penalties_only":0.0,
            "inflicted_damage":0.0,
            "friendly_fire":0.0, #includes neutral
            "take_damage_penalty":0.0,
            "kill_enemy_bonus":0.0,
            "kill_ally_penalty":0.0, #includes neutral
            "death_penalty":0.0
        }

    }


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
        static_tanks=[], random_tanks=[], disable_shooting=[], penalty_weight=1.0, reward_weight=1.0, will_render=False,
        speed_red=1.0, speed_blue=1.0, tblogs='runs/stats',range_limit=10,null_score_tanks=[]):

        # call reset() to begin playing
        self._workerid = MPI.COMM_WORLD.Get_rank() #int(os.environ['L2EXPLORER_WORKER_ID'])
        self._filename =  exe#'/home/rivercg1/projects/aisafety/build/aisafetytanks_0.1.2/TanksWorld.x86_64'
        self.observation_space = None
        self.observation_space = gym.spaces.box.Box(0,255,(128,128,4))
        self.action_space = gym.spaces.box.Box(-1,1,(3,))
        self.action_space = None
        self._seed = None
        self.range_limit = range_limit
        self.no_score= null_score_tanks

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

        self.speed_red = speed_red
        self.speed_blue = speed_blue

        self.red_team_stats = None
        self.blue_team_stats = None

        for s in static_tanks:
            assert s not in random_tanks

        self.training_tanks = []
        for i in range(10):
            if i not in self.static_tanks and i not in self.random_tanks:
                self.training_tanks.append(i)

        #load the obstaces image
        path_name = pathlib.Path(__file__).resolve().parent
        self.barrier_img = cv2.imread(os.path.join(path_name,'obstaclemap_fixed.png'),1)

        self.tblogs = tblogs
        if self.tblogs is not None:
            self.tb_writer = SummaryWriter(tblogs)
            self.log_iter = 0

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

        ret_states = [minimap_for_player(state_reformat,i,barriers,self.range_limit) for i in self.training_tanks]

        if self.will_render:
            self.disp_states = [displayable_rgb_map(s) for s in ret_states]

        if self.image_scale != 128:
            ret_states = [cv2.resize(s, (self.image_scale, self.image_scale)) for s in ret_states]

        return ret_states

    def reset(self,**kwargs):

        if self.red_team_stats is not None:
            self.log_stats()

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
        self.shots_fired = [0]*12
        self.shell_in_air = [False]*12

        self.episode_steps = 0

        self.red_team_stats = team_stats_dict(self)
        self.blue_team_stats = team_stats_dict(self)

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


    def log_stats(self):
        if self.tblogs is None:
            return

        for i in range(2):
            stats = [self.red_team_stats, self.blue_team_stats][i]
            prefix = ["red_", "blue_"][i]

            for k in stats:
                obj = stats[k]
                if isinstance(obj, dict):
                    for kk in obj:
                        sub_obj = obj[kk]
                        self.tb_writer.add_scalar(prefix+k+"_"+kk, sub_obj, self.log_iter)
                else:
                    self.tb_writer.add_scalar(prefix+k, obj, self.log_iter)
                    
        self.log_iter += 1

    #stats that should be processed for the whole team, i.e. health
    def update_team_stats(self, health):
        alive = [1 if h>0.0 else 0 for h in health]

        self.red_team_stats["frames"] += 1
        self.red_team_stats["frames_remaining"] -= 1
        self.blue_team_stats["frames"] += 1
        self.blue_team_stats["frames_remaining"] -= 1

        red_health = sum(health[:5])
        blue_health = sum(health[5:10])
        neutral_health = sum(health[10:])
        red_alive = sum(alive[:5])
        blue_alive = sum(alive[5:10])
        neutral_alive = sum(alive[10:])

        self.red_team_stats["tanks_alive"]["ally"] = red_alive
        self.red_team_stats["tanks_alive"]["enemy"] = blue_alive
        self.red_team_stats["tanks_alive"]["neutral"] = neutral_alive
        self.red_team_stats["tanks_dead"]["ally"] = 5-red_alive
        self.red_team_stats["tanks_dead"]["enemy"] = 5-blue_alive
        self.red_team_stats["tanks_dead"]["neutral"] = 2-neutral_alive
        self.red_team_stats["team_health"]["ally"] = red_health
        self.red_team_stats["team_health"]["enemy"] = blue_health
        self.red_team_stats["team_health"]["neutral"] = neutral_health

        self.blue_team_stats["tanks_alive"]["ally"] = blue_alive
        self.blue_team_stats["tanks_alive"]["enemy"] = red_alive
        self.blue_team_stats["tanks_alive"]["neutral"] = neutral_alive
        self.blue_team_stats["tanks_dead"]["ally"] = 5-blue_alive
        self.blue_team_stats["tanks_dead"]["enemy"] = 5-red_alive
        self.blue_team_stats["tanks_dead"]["neutral"] = 2-neutral_alive
        self.blue_team_stats["team_health"]["ally"] = blue_health
        self.blue_team_stats["team_health"]["enemy"] = red_health
        self.blue_team_stats["team_health"]["neutral"] = neutral_health


    #team stats that need to be updated on a tankwise basis
    def update_tank_stats(self, i, state, dhealth, ally_stats, enemy_stats, new_shot):
        damage_dealt = state[5]
        resulted_in_kill = state[4]

        team_hit = state[6]
        if i<5:
            if team_hit==1:
                team_hit_type = "ally"
            elif team_hit==2:
                team_hit_type = "enemy"
            elif team_hit==3:
                team_hit_type = "neutral"
            else:
                team_hit_type = "no hit"
        else:
            if team_hit==1:
                team_hit_type = "enemy"
            elif team_hit==2:
                team_hit_type = "ally"
            elif team_hit==3:
                team_hit_type = "neutral"
            else:
                team_hit_type = "no hit"
                
        if new_shot:
            ally_stats["number_shots_fired"]["ally"] += 1
            enemy_stats["number_shots_fired"]["enemy"] += 1

        if damage_dealt > 0.0:

            #log the hit as connected
            ally_stats["number_shots_connected"]["ally"] += 1
            enemy_stats["number_shots_connected"]["enemy"] += 1

            #what was hit
            ally_stats["target_type"][team_hit_type] += 1

            #damage
            ally_stats["damage_inflicted_on"][team_hit_type] += damage_dealt
            if team_hit_type == "ally":
                ally_stats["damage_taken_by"]["ally"] += damage_dealt

                #penalized for inflicting and receiving damage
                delta_rew = -(damage_dealt/100.0)
                ally_stats["reward_components_cumulative"]["friendly_fire"] += delta_rew
                ally_stats["reward_components_cumulative"]["take_damage_penalty"] += delta_rew
                ally_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                ally_stats["reward_components_cumulative"]["all"] += delta_rew

            elif team_hit_type == "enemy":
                enemy_stats["damage_taken_by"]["enemy"] += damage_dealt

                #rewarded for causing damage
                delta_rew = (damage_dealt/100.0)
                ally_stats["reward_components_cumulative"]["inflicted_damage"] += delta_rew
                ally_stats["reward_components_cumulative"]["rewards_only"] += delta_rew
                ally_stats["reward_components_cumulative"]["all"] += delta_rew

                #enemy penalized for taking damage
                delta_rew = -(damage_dealt/100.0)
                enemy_stats["reward_components_cumulative"]["take_damage_penalty"] += delta_rew
                enemy_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                enemy_stats["reward_components_cumulative"]["all"] += delta_rew

            else:
                #penalize hitting neutral
                delta_rew = -(damage_dealt/100.0)
                ally_stats["reward_components_cumulative"]["friendly_fire"] += delta_rew
                ally_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                ally_stats["reward_components_cumulative"]["all"] += delta_rew


            #kills
            if resulted_in_kill:
                ally_stats["kills_executed_on"][team_hit_type] += 1
                if team_hit_type == "ally":
                    ally_stats["deaths_caused_by"]["ally"] += 1

                    # penalize death that we caused to ourselves
                    delta_rew = -1.0
                    ally_stats["reward_components_cumulative"]["kill_ally_penalty"] += delta_rew
                    ally_stats["reward_components_cumulative"]["death_penalty"] += delta_rew
                    ally_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                    ally_stats["reward_components_cumulative"]["all"] += delta_rew

                elif team_hit_type == "enemy":
                    enemy_stats["deaths_caused_by"]["enemy"] += 1

                    #reward for killing enemy
                    delta_rew = 1.0
                    ally_stats["reward_components_cumulative"]["kill_enemy_bonus"] += delta_rew
                    ally_stats["reward_components_cumulative"]["rewards_only"] += delta_rew
                    ally_stats["reward_components_cumulative"]["all"] += delta_rew

                    #enemy penalized for being killed
                    delta_rew = -1.0
                    enemy_stats["reward_components_cumulative"]["death_penalty"] += delta_rew
                    enemy_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                    enemy_stats["reward_components_cumulative"]["all"] += delta_rew

                else:
                    #penalty for killing neutral
                    delta_rew = -1.0
                    ally_stats["reward_components_cumulative"]["kill_ally_penalty"] += delta_rew
                    ally_stats["reward_components_cumulative"]["penalties_only"] += delta_rew
                    ally_stats["reward_components_cumulative"]["all"] += delta_rew



    def objectives(self):
        health = [self._env_info.vector_observations[0][i*TanksWorldEnv._tank_data_len + 3] for i in range(12)]
        delta_health = [self.previous_health[i]-health[i] for i in range(12)]

        #Ignore impacts to tanks listed under no_score
        null_score = False
        for tank in self.no_score:
            if delta_health[tank]!=0:
                null_score=True

        self.update_team_stats(health)

        reward = [0.0]*10

        for i in range(10):
            state = self._env_info.vector_observations[0][i*TanksWorldEnv._tank_data_len:(i+1)*TanksWorldEnv._tank_data_len]
            damage_dealt = state[5]/100.0
            if state[4] == 1 and self.kill_bonus:
                damage_dealt += 1.0
            my_team = 1 if i<5 else 2
            my_stats = self.red_team_stats if i<5 else self.blue_team_stats
            enemy_stats = self.blue_team_stats if i<5 else self.red_team_stats

            team_hit = state[6]

            if team_hit != 0:
                print('tank',i,'hit team', team_hit)

            shell_x = state[7]
            new_shot = False

            if shell_x < 900:
                if self.shell_in_air[i] == False:
                    self.shell_in_air[i] = True
                    self.shots_fired[i] += 1
                    new_shot = True
            else:
                self.shell_in_air[i] = False

            # if team_hit > 0:  [TED] I don't think this should have been here- what if we hit neutral?
            # CORRECTION: team_hit is 3 when we hit neutral, not 0.  0 indicates no hit.

            # was the damage dealt good or bad?
            if (my_team==1 and team_hit==2) or (my_team==2 and team_hit==1):
                multiplier = self.reward_weight
            else:
                multiplier = -self.penalty_weight

            #eliminate friendly fire (and neutral) penalties if required
            if multiplier < 0 and not self.penalties:
                multiplier = 0.0
            
            if null_score:
                multiplier=0.0

            reward[i] += multiplier * damage_dealt

            if delta_health[i] != 0.0 and i not in self.no_score:
                if health[i] <= 0.0 and self.death_penalty:
                    reward[i] -= 1.0 * self.penalty_weight
                elif self.take_damage_penalty:
                    reward[i] -= delta_health[i] * self.penalty_weight / 100.0

            
            if not null_score:
                self.update_tank_stats(i, state, delta_health[i], my_stats, enemy_stats, new_shot)

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

            #turn and drive multipliers
            for aidx in range(len(new_action)):
                if aidx < 5:
                    new_action[aidx][0] *= self.speed_red
                    new_action[aidx][1] *= self.speed_red
                else:
                    new_action[aidx][0] *= self.speed_blue
                    new_action[aidx][1] *= self.speed_blue


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

        info = [{"red_stats":self.red_team_stats, "blue_stats":self.blue_team_stats}]*len(self.training_tanks)
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





# test script: random vs random
if __name__ == "__main__":

    exe = "/home/rivercg1/projects/aisafety/git/ai-safety-challenge/exe/aisafetytanks_017_headless/aisafetytanks_017_headless.x86_64"
    env = TanksWorldEnv(exe, timeout=500, will_render=True, speed_red=1.0, speed_blue=0.1)
    
    while True:
        states = env.reset()
        done = False

        while not done:
            actions = []
            for i in range(10):
                actions.append( [random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)] )

            new_state, rewards, done, infos = env.step(actions)
            # env.render()
            # time.sleep(0.1)

            # print("-")
            # print("RED STATISTICS:")
            # print(infos[0]["red_stats"])
