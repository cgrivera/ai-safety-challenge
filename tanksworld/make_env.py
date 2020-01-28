

from tanksworld.env import TanksWorldFPVEnv
import my_config as cfg

def make_env():
	return TanksWorldFPVEnv(cfg.args.exe)
