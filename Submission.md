Quick Links: [Overview & Installation](./README.md) | [Environment](./TanksWorldData.md) | [Evaluation](./Evaluation.md) | [Submission](./Submission.md) | [AI Arena](https://gitlab.jhuapl.edu/staleew1/ai-arena-v5/tree/master/)

# Submission Format and Rules

Submitting a model for the AI Safety Challenge has its own rules and interface requirements.  Most important are the following points, which will be explained in more detail further below:

- Each submission should be a python package that can be installed via pip
- There is a special file structure and interface for submissions (the example in the tanksworld/ directory is NOT a valid submission)
- Each submission will provide the means to control a single tank, and five copies of your submission will be run to control a whole team
- Communication amongst tanks is not under your control.  It is handled exclusively by the environment.
- Submissions must behave in the spirit of the competition


## Submissions are python 3 packages
Special considerations are necessary to evaluate two submissions against each other (or one submission against a baseline), if the two submissions have different or conflicting software environments.  Our solution to this is to run each submission inside its own isolated conda environment, and have these talk to each other via REDIS.  This communication is not something you need to handle.  You should specify in your setup.py file which external python libraries you need. They will be pulled in when your submitted package is installed.

![diagram](./conda_envs.png)

### Why python packages?
To reiterate: Python packages can express in their setup.py file what other python libraries they require to run.  This allows you to not only submit your policy, but also a list of packages that your policy needs, such as specific versions of tensorflow, pytorch, keras, numpy, pandas, etc.  You package will be locally installed into the isolating conda environment.

### NOTE: If you do NOT specify what packages you need, we will assume your submission is pure python.


## Submissions have their own python interface and format

Submissions should consist of a python package which at the very least contains a file called policy.py that contains a class called Policy.  For the file structure of submissions please see the submisison template or the example submission.

The interface for the policy class is very simple.  Besides a constructor, you should implement the following methods:

````python
def game_reset(self, tank_ID):
	# This is called when a new game starts, and provided a unique ID for the tank that will be controlled.  
	# Use it if you needed it, otherwise just have it pass.
	# No return value.


def get_actions(self, state):
	# Given a tank's image state, reply with an action for that tank to take.
	# returns: a list of three floats in range [-1.0, 1.0] which constitute a single action.

````

## Creating different behaviors for different members of your tanksworld team
Because the tank_id is provided in the interface you can use that information to create different roles for different teammembers.  Be aware though that you should not count on any individual tank id on your team being under your control.  An example of this would be executing one policy for tanks 0-2 and another policy for 3-4.


````python
def game_reset(self, tank_ID):
	# This is called when a new game starts, and provided a unique ID for the tank that will be controlled.  
	# Use it if you needed it, otherwise just have it pass.
	# No return value.
  self.tank_id = tank_ID
  self.policy_modelA = PolicyA()
  self.policy_modelB = PolicyB()


def get_actions(self, state):
	# Given a tank's image state, reply with an action for that tank to take.
	# returns: a list of three floats in range [-1.0, 1.0] which constitute a single action.
  if self.tank_id < 3:
    return self.policy_modelA(state)
  else:
    return self.policy_modelB(state)

````


## Submissions control ONE tank, not a whole team

### NOTE: Some of this information has changed from what was disseminated early in the competition period.

Your submission will be used to control one tank, and should be structured accordingly.  It will be duplicated by the evaluation software in order to control the entire team in a decentralized manner.  Each tank has a unique ID which will be given to your policy, in case you wish to provide specialized roles based on the tank ID.  You will not have control over which tanks are placed where in the game initialization.

This setup has several pros and cons:
#### Pros:
- Team size can be variable
- Humans can take over the role of a tank and play alongside an AI
- Submissions can be run alongside other teammates that have custom behaviors
- Encourages decentralized approaches

#### Cons:
- Centralized control is not possible during evaluation
- No customized communication between teammates (see next section)

![diagram](./submission_copies.png)


## Clarifications on Tank-Tank Communication

Early in the competition it was mentioned that tanks will be communicating with each other.  Please note that this is entirely handled by the environment.

The images that are provided to a tank currently reflect a large radius of perception for that tank.  In the future, this radius may be limited, and entities that are far away will only be observable if (1) an ally can observe them and (2) that ally is close to the current tank (or close to another tank that is, in turn, close to this tank, etc.).

Again, this is all handled by the environment and will manifest as changes to the imagery visible to the tanks.  There is no expectation that particpants will design communication methods between tanks.

Furthermore, tanks on the same team should NOT communicate with each other in evaluation.  You policy submission should not utilize class variables or communication libraries to talk to other instaniated policies.  We are going to be able to look at your code.  No sneaky stuff.  See below.


## Spirit of the competition
This is a purposefully vague heading that basically means "no funny business".  This is a non-exhaustive list of things that your submission should NOT do:

- Connect to the internet to communicate with external servers or compute resources
- Take over most of the compute resources of the evaluation machine (CAUTION: This is the default behavior of tensorflow)
- Run a REDIS server or attempt to communicate directly with the evaluation REDIS server
- Attempt to communicate extra information to teammates that is not provided by the environment
- Crash the evaluaton machine or do weird system-level stuff
- Exploit bugs that you have found in the game and decided not to report
