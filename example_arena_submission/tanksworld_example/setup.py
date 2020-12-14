from setuptools import setup, find_packages

setup(
    name='tanksworld_example',
    version='1.0.0',
    author='Ted Staley',
    author_email='admin@aisaeftychallenge.edu',
    description='AI Safety Tanksworld Example Submission',
    packages=find_packages(),
    install_requires=['numpy','stable_baselines==2.9.0','tensorflow==1.14.0'],
)
