"""
Setup file
"""
from setuptools import setup

VERS = {}
with open("./spyeeg/version.py") as fp:
    exec(fp.read(), VERS)

setup(
    name='sPyEEG',
    version=VERS['__version__'],
    packages=['spyeeg'],
    url='',
    license='',
    author='Pierre Guilleminot & Mikolaj Kegler',
    description='Package for modelling EEG responses to speech.'
)
