from setuptools import setup, find_packages

setup(
    name='tanksworld_example',
    version='1.0.0',
    url='https://gitlab.jhuapl.edu/rivercg1/ai-safety-challenge.git',
    author='Ted Staley',
    author_email='edward.staley@jhuapl.edu',
    description='AI Safety Tanksworld Example Submission',
    packages=find_packages(),
    install_requires=['numpy','stable_baselines==2.9.0','tensorflow==1.14.0'],
)
