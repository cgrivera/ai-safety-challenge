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

#this is a list of assignments of entity <---> policy
match_list = [[1,1,1,1,1]]

#for each policy above, what type of policy is it
#you can specify a string name or TODO: a path to a custom algo
policy_types = {1:"ppo"}

#train with this configuration
while True:
    arena.kickoff(match_list, policy_types, 1500000, scale=True, render=True)
