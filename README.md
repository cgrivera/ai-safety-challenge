# AI Safety Challenge (TanksWorld)

Home project for AI Arena's TanksWorld challenge environment.

Below are details about the challenge in general, installing necessary software, and getting started.  

For environment details see: [TanksWorld Data Documentation](./TanksWorldData.md) 

For rules and evaluation see: [TODO_EVALUATION_SCHEME_DOCS](./Evaluation.md)

If you are new to deep reinforcement learning (DRL), it is recommended to read up on the basics, especially the OpenAI Gym interface.  See the [AI Arena Documentation](https://gitlab.jhuapl.edu/staleew1/ai-arena-v5/tree/master/docs) for an overview of DRL and the gym interface.

## Installation Quick Start Guide
These install instructions are for Linux clients.  We recommend installing the anaconda distribution of python to manage the dependencies.

#### Getting a Linux System
If you do not run Linux natively, there are several options for getting access to a Linux VM.  The easiest is probably to use myAPLCloud (vanity url myaplcloud/)
Make sure you (1) request a Linux machine, preferably Ubuntu 18.04, and (2) request intranet access, not DMZ, so you can access gitlab.

More information about myAPLCloud can be found here:
https://aplweb.jhuapl.edu/services/Pages/APLCloud.aspx


#### Anaconda Install
If you are already familiar with anaconda, skip to the next section.

Visit https://www.anaconda.com/distribution/ and download the python 3 version.  This should download a .sh script.
Navigate to your home folder in a terminal and run the script:

```` sh
bash <path/to/anaconda/script.sh>
````

When prompted, answer yes.  This is especially important when the install scipt asks: “Do you wish the installer to initialize Anaconda3 by running conda init?”.
Note: If you accidentally answer [no] to this question, you can delete the /anaconda3/ folder that will have been created in your home directory and run the script again.

Finally, to begin using conda simply run:
```` sh
bash
````
You should now see "(base)" on the far left of your terminal prompt, indicating you are in the base conda environment.  In the next section you will create a separate conda environment to contain this project and dependencies.

#### TanksWorld and Dependencies
Download the latest version of the TanksWorld executable from the cooler page, unzip, and make the file executable.  https://cooler.jhuapl.edu/file/group/354691/all
```` sh
unzip <latest_aisafety_build.zip>
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

Go into the tankworld folder and run the main script with mpi.  The -n flag indicates the number of processes that will run simultaneously. Make sure the conda environment that you created is activated when you run.
 - exe is the absolute path to the latest tanksworld executable
 - logdir is the location that logfiles, saved policies, and performance plots will be saved.
 - n is the number of processes, if you add enough processes, additional environments will be run
```` sh
cd ai-safety-challenge/tanksworld
mpiexec --oversubscribe -n 14 python my_main_script.py --exe /absolute/path/to/the/executable --logdir testrun
````


## TanksWorld Details

### Environment Description

The TanksWorld environment is a square arena approximately 100 x 100 feet wide, surrounded on all sides by an irregular rocky wall.  
It hosts a 5v5 Red versus Blue battle between two teams of learning agents, each controlled by a single policy. 
Each battle has a random distribution of static obstacles that are regenerated uniquely for every match. 
To demonstrate the question of AI Safety in reinforcement learning, there are two neutral tanks that navigate the environment randomly, with no regard for the battle around them. They can be killed, but cannot attack. 

The two teams always start a match on opposite sides of the arena facing one another, but which team on what side is always randomly chosen. 
Each tank can drive and shoot. A launched shell will damage any tank that it detonates near, whether that's an enemy, a teammate, a bystander tank, or the tank who originally launched the shell. 
Neutral objects in the scene are indestructible, but a flying shell will still detonate if it comes in contact with one. 

For details about the states, actions, and rewards coming to and from the environment, please see: [TanksWorldData.md](./TanksWorldData.md)


## AI Arena

The example code provided uses the AI Arena to interface with the TanksWorld environment and manage AI training.  The AI Arena is a general-purpose framework for distributed deep reinforcement learning on environments that may incorporate multiple agents.  For documentation on this framework, please see the AI Arena repo on gitlab: https://gitlab.jhuapl.edu/staleew1/ai-arena-v5

In particular, look at the documentation located at: https://gitlab.jhuapl.edu/staleew1/ai-arena-v5/tree/master/docs

Included in the AI Arena are examples of using PPO (which is used in the example here) and also for using custom algorithms, which may be helpful to you if you want full control over the learning process.  The example of a custom policy can be found here: https://gitlab.jhuapl.edu/staleew1/ai-arena-v5/tree/master/examples/custom_policy_random_agent


## Underlying Unity Simulation

### NOTE: This has been moved to: [UnitySim.md](./UnitySim.md)