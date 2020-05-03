from tanksworld.env import TanksWorldEnv
import importlib
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='AI Safety TanksWorld')
    parser.add_argument('--exe', help='the absolute path of the tanksworld executable')
    parser.add_argument('--package_name', help='the package name for the submission')
    args = parser.parse_args()

    env =  TanksWorldEnv(args.exe,
			action_repeat=6,                        # step between decisions, will be 6 in evaluation
			image_scale=128,            # image size, will be 128 in evaluation
			timeout=500,                            # maximum number of steps before episode forces a reset
			friendly_fire=True,            # do you get penalized for damaging self, allies, neutral
			take_damage_penalty=True,   # do you get penalized for receiving damage (double counts w/ self-freindly-fire)
			kill_bonus=True,                  # do you get +1 for killing enemy (-1 penalty for friendly fire kills if friendly fire is on)
			death_penalty=True,                    # do you get -1 for dying
			static_tanks=[],                      # indices of tanks that do not move (not exposed externally, changes number of controllable players)
			random_tanks=[1,2,3,4,5,6,7,8,9],      # indices of tanks that move randomly (not exposed externally, changes number of controllable players)
			disable_shooting=[1,2,3,4,5,6,7,8,9],              # indices of tanks that cannot shoot (i.e. to allow random movement without shooting)
			reward_weight=1.0,
			penalty_weight=1.0,
			will_render=True)

    mymodule = importlib.import_module(args.package_name+'.policy')
    #from mymodule import Policy
    policy = mymodule.Policy()

    done = False
    state = env.reset()
    while not done:
        action = policy.get_actions(state[0])
        state,reward,done,info = env.step([action])
    print("Test completed successfully!")
