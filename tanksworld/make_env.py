

from tanksworld.env import TanksWorldEnv
import my_config as cfg

def make_env():
	return TanksWorldEnv(cfg.args.exe)
