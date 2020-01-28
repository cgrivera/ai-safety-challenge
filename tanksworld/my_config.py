import os
from make_env import make_env
import gym
import os
import argparse

parser = argparse.ArgumentParser(description='AI Safety TanksWorld')
parser.add_argument('--logdir',help='the location of saved policys and logs')
parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
args = parser.parse_args()

# Tell the arena where it can put log files that describe the results of
# specific policies.  This is also used to pass results between root processes.

LOG_COMMS_DIR = "logs/"+args.logdir
#os.makedirs(LOG_COMMS_DIR, exist_ok=True)

# Define where to find the environment
# if make_env is in this directory, os.getcwd() will suffice

MAKE_ENV_LOCATION = os.getcwd()


# Tell the arena what observation and action spaces to expect

#temp = make_env()

OBS_SPACES = [gym.spaces.box.Box(0,255,(128,128,4))]*10
ACT_SPACES = [gym.spaces.box.Box(-1,1,(3,))]*10
