from setuptools import setup, find_packages

setup(
    name='tanksworld',
    version='1.0.0',
    url='https://gitlab.jhuapl.edu/rivercg1/ai-safety-challenge.git',
    author='Corban Rivera',
    author_email='corban.rivera@jhuapl.edu',
    description='AI Safety Tanksworld',
    packages=find_packages(),
    install_requires=['numpy','mlagents'],
)
