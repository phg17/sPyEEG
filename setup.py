"""
Setup file
"""
import setuptools

VERS = {}
with open("./spyeeg/version.py") as fp:
    exec(fp.read(), VERS)

setuptools.setup(
    name='spyeeg',
    version=VERS['__version__'],
    packages=setuptools.find_packages(),
    license='MIT',
    author='Pierre Guilleminot & Mikolaj Kegler',
    description='Package for modelling EEG responses to speech.'
)