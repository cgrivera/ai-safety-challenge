

from tanksworld.env import TanksWorldStackedEnv
import my_config as cfg

def make_env():
	return TanksWorldStackedEnv(cfg.args.exe)
