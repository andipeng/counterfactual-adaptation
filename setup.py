import os
import sys
from setuptools import setup, find_packages

print("Installing aligning construals")

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='aligning_construals',
    description='aligning human and robot construals',
    long_description=read('README.md'),
    author='Andi Peng',
)