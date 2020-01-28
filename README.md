
Home project for AI Arena's Tanks! challenge environment.

### Installation Quick Start Guide
These install instructions are for linux clients.  We recommend installing the anaconda distribution of python to manage the dependencies

Download the latest version of the TanksWorld executable from the cooler page, unzip, and make the file executable.  https://cooler.jhuapl.edu/file/group/354691/all
```` sh
https://cooler.jhuapl.edu/serve-file/e1580171309/l1580164104/da/c1/qty87IAnTo2U0FUU8-_GD715POaKi4WFbL40PjobH5E/260000/260137/file/1580164104aisafetytanks_0.1.2.zip
unzip 1580164104aisafetytanks_0.1.2.zip
chmod -R 755 aisafetytanks
````

Create a project conda environment and install some requirements
````sh
conda create -n aisafety python=3.6
conda activate aisafety
conda install -c conda-forge tensorflow=1.14.0 mpi4py
pip install mlagents==0.9.3 stable_baselines
````

Clone the ai safety challenge and the AI arena repo and install.
```` sh
git clone https://gitlab.jhuapl.edu/rivercg1/ai-safety-challenge.git
git clone https://gitlab.jhuapl.edu/staleew1/ai-arena-v5.git
pip install -e ai-safety-challenge/
pip install -e ai-arena-v5/
````

Go into the tankworld folder and run the main script with mpi.  The -n flag indicates the number of processes that will run simultaneously. 
 - exe is the absolute path to the latest tanksworld executable
 - logdir is the location that logfiles, saved policies, and performance plots will be saved.
 - n is the number of processes, if you add enough processes, additional environments will be run
```` sh
cd ai-safety-challenge/tanksworld
mpirun -n 13 --oversubscribe python my_main_script.py --exe /absolute/path/to/the/executable --logdir testrun
````




## TanksWorld Details

### Game Controller -> Key Code
Note that the joysticks and d-pad of the game controller are not "keys" and so have no keycode (that I know of).
JoystickButton_:

0. A
1. B
2. X
3. Y
4. Right bumper
5. Left bumper
6. Start (right-hand side)
7. Back (left-hand side)
8. Click left-stick
9. Click right-stick

### Features

* Randomized starting positions - an entire team always appears on one side, facing towards the other side.
* Fixed number of teams and teammates - 5v5 red vs blue
* Two neutral "green team" tanks that navigate randomly and do not attack.


## Ins

### Vector action

3 floats, all range -1 to 1:

1. forward/backward acceleration
2. angular left/right acceleration
3. shoot/don't shoot ( >0.5 is shoot, <= 0.5 is no-op)

### Decision Interval

The decision interval for the tanks is every 6 steps. This means a learner can only choose an action every 6 steps, and for the steps between decisions, the last known action is applied each time. If a decision is to turn left, it will continue turning left for the next 6 steps.

The vector observation is sent every 6 steps too. Between decisions, nothing new is sent.

### Reset parameters

None right now...could be added to specify stuff like:

- number of bystander tanks
- density of obstacles
- tank stats like damage per shell, speed, cooldown timer length, etc...

## Outs

### Vector Observation

float array of length 84

Every agent reads and returns the exact same observation: truth data about all active tanks.

The observation is the states of 12 tanks (5 red, 5 blues, 2 neutral) one after the other. The state of one tank is a float array of length of 7.

In order, the states describe red tanks 1-5, blue tanks 1-5, neutral tank 1, neutral tank 2.

The state of one tank looks like this:

0. position x, (appx -40 to 40)
1. position y, (appx -40 to 40)
2. rotation degrees, 0 to 360.... 0 is facing towards +z direction; 90 is facing in the +x direction, etc
3. My HP (100 to 0)
4. how much damage I dealt to anyone/anything this frame, (0 to 100)
5. the team of the thing I hit this frame (0 for no hits occured, 1 for team 1 (red), 2 for team 2 (blue), 3 for a neural bystander tank (uh oh!))
6. Whether I killed someone this frame (1 for yes, 0 for no kills)

e.g. a state of a tank at the start of the game might look like [30, 20, 180, 100, 0, 0, 0]

and the complete observation would be twelve of those, one after the other, in a long array

### Visual Observation

- one FPV at 84x84 px RGB
- one Minimap view at 84x84 px RGB, which is identical for all agents. It represents the position & rotation of all active tanks.
	- 5 and 5 isoceles triangles with red/blue color to represent tank position and rotation.
	- two green triangles to represent bystander tanks
	- little circles represent active shells; the color corresponds to the team of the tank which launched it.

### Debug Rewards

The brain returns some debug rewards that I made- rewards tanks for moving fast and hitting enemies, and penalizes them for hurting friends & bystanders.
