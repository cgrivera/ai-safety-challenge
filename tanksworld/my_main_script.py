# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
# To create a script for the arena, we need to interact with a UserStem object.
# This object is active only on the root process, and broadcasts commands to the
# other processes.

import math, time, random

from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
import argparse


import my_config as cfg

arena = make_stem(cfg.MAKE_ENV_LOCATION, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)

# --- only the root process will get beyond this point ---

# the first 5 players in the gamew will use policy 1
match_list = [[1,1,1,1,1]]

# policy 1 is PPO
policy_types = {1:"ppo"}

#kwargs to configure the environment
kwargs_1 = {"static_tanks":[], "random_tanks":[5,6,7,8,9], "disable_shooting":[],"friendly_fire":False}

# run each copy of the environment for 300k steps
arena.kickoff(match_list, policy_types, 300000, scale=True, render=False, env_kwargs=kwargs_1)
