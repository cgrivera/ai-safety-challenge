
# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
from tanksworld.env import TanksWorldEnv
import tanksworld.my_config as cfg

# example of passing in kwargs and using them
def make_env(friendly_fire=True, take_damage_penalty=True, kill_bonus=True, death_penalty=True,
	static_tanks=[], random_tanks=[], disable_shooting=[], reward_weight=1.0, penalty_weight=1.0,range_limit=10, null_score_tanks =[], **kwargs):

	return TanksWorldEnv(cfg.args.exe,
		action_repeat=6, 			# step between decisions, will be 6 in evaluation
		image_scale=128,            # image size, will be 128 in evaluation
		timeout=500,				# maximum number of steps before episode forces a reset
		friendly_fire=friendly_fire, 		# do you get penalized for damaging self, allies, neutral
		take_damage_penalty=take_damage_penalty,   # do you get penalized for receiving damage (double counts w/ self-freindly-fire)
		kill_bonus=kill_bonus, 			# do you get +1 for killing enemy (-1 penalty for friendly fire kills if friendly fire is on)
		death_penalty=death_penalty,			# do you get -1 for dying
        static_tanks=static_tanks, 			# indices of tanks that do not move (not exposed externally, changes number of controllable players)
        random_tanks=random_tanks,	# indices of tanks that move randomly (not exposed externally, changes number of controllable players)
        disable_shooting=disable_shooting, 		# indices of tanks that cannot shoot (i.e. to allow random movement without shooting)
        reward_weight=reward_weight,
        penalty_weight=penalty_weight,
        will_render=True,
		range_limit =range_limit,
		null_score_tanks=null_score_tanks)		#tank indices that don't contribute to the reward or statistics
		# prepare rgb images for displaying when render() is called.  If not rendering turn off.

		# NOTE: Make sure if you set static_tanks or random_tanks, or adjuist image_scale, that you make appropriate changes in my_config.py!!!
		# If you are making a curriculum that changes the number of tanks, you will probably need to keep all of them turned on and instead control them
		# with a policy.
