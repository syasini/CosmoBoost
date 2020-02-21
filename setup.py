from setuptools import setup

with open("requirements.txt", "r") as f:
    reqs = [line.rstrip("\n") for line in f if line != "\n"]

setup(name='cosmoboost',
      version='1.0',
      description='a python package for boosting the cosmos!',
      url='https://github.com/syasini/CosmoBoost',
      install_requires=reqs,
      author='Siavash Yasini',
      author_email='yasini@usc.edu',
      license='MIT',
      packages=['cosmoboost'])
