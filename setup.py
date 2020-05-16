from setuptools import setup, find_packages

# read the contents of requirements.txt
with open("requirements.txt", "r") as f:
    reqs = [line.rstrip("\n") for line in f if line != "\n"]

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='cosmoboost',
      version='1.1.6',
      description='a python package for boosting the cosmos!',
      url='https://github.com/syasini/CosmoBoost',
      install_requires=reqs,
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Siavash Yasini',
      author_email='yasini@usc.edu',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      )
