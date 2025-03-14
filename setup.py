"""
Setup file
"""
import setuptools

setuptools.setup(
    name='spyeeg',
    version= "0.1.0",
    packages=setuptools.find_packages(),
    license='BSD 3',
    author='Pierre Guilleminot & Mikolaj Kegler',
    description='Package for modelling EEG responses to stimuli.'
)