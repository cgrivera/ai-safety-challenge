# ©2020 Johns Hopkins University Applied Physics Laboratory LLC.
from setuptools import setup, find_packages

setup(
    name='tanksworld',
    version='1.0.0',
    author='Corban Rivera',
    author_email='admin@aisafetychallenge.edu',
    description='AI Safety Tanksworld',
    packages=find_packages(),
    install_requires=['numpy','mlagents==0.9.3'],
)
