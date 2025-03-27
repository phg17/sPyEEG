# sPyEEG

#### Version: 0.2.2

Package for modelling s/M/EEG responses to stimuli. In other words, for mapping sensory or cognitive features, through python (*sPyeech*) to EEG (*sPyEEG*)... and the other way around! 

Not *mind-reading* for espionage purposes ;). (Definitely not that)

### Setup

#### Requirements
Package builds on top on [MNE](https://mne.tools/stable/index.html) and relies on a similar set of dependencies and 3rd party packages listed in ```requirements.txt```. Normally, installing via pip will take care of dependencies but in case, you can check that there won't be problems.

#### Installation
For a standard installation (but this will require to be installed if you need to install another version of the library):

```bash
pip install spyeeg
```

#### Citation
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7006933.svg)](https://doi.org/10.5281/zenodo.7006933)

##### Tested on:
- macOS Big Sur v11.1
- Ubuntu 18.04.5 LTS
- Windows 10 22H2

### Modules (sketch)
- **models** - for all your modelling needs
  - TRF: Temporal Response Function a.k.a Ridge regression a.k.a. fancy linear regression, optimized for speed
  - iRRR: integrative reduced rank regression a.k.a fancier linear regression
  - _methods: useful methods used by several model classes
  - tbc

#### Contributors:
- Pierre Guilleminot (pierre.hieu.guilleminot@gmail.com)
- Mikolaj Kegler (mak616@ic.ac.uk)
- Michael Thornton (m.thornton20@imperial.ac.uk)

Last updated: 15th March 2025
