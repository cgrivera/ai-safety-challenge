### Example submission template for AI Arena based submissions.

Example submission templates have been added to the AI Safety Challenge repo.  There is a template for those of you who have trained through the AI Arena and produced a model weights file (by default this is called ppo_save.zip).  An example network weights file was also added to submission example.  There is also a basic template to illustrate the minimum required elements for sumissiont.  

Be sure to change the name of your python package in setup.py to match your team name or at least something unique.  

````python

setup(
    name='tanksworld_example', # <--- Your teamname goes here, no spaces, this should be unique, and match the directory name in the same directory
    version='1.0.0',
    url='https://gitlab.jhuapl.edu/rivercg1/ai-safety-challenge.git',
    author='Ted Staley',   #<--  Add the authors name
    author_email='edward.staley@jhuapl.edu',   #<-- email address
    description='AI Safety Tanksworld Example Submission',  #<-- flavor text, add anything
    packages=find_packages(),
    install_requires=['numpy','stable_baselines==2.9.0','tensorflow==1.14.0'], #<-- if you have specific dependencies for your submission add them here
)
````

### Zipping up the submission
There should be a single directory at the top level.  The directory structure should look as follows.  You can include additional python files if needed.  

```` sh
tanksworld_example <-- your submission zip file should include a single top level directory
    ├── setup.py   <-- required, update this file as described above
    ├── tanksworld_example   <-- required, this directory name needs to match your package name
    │   ├── __init__.py   <-- required
    │   ├── policy.py   <-- required, keep the name policy.py
    │   ├── ppo_save.zip  <-- this is an example saved weights file
````