# sPyEEG

#### Version: 0.0.1

Package for modelling EEG responses to speech. In other words, for mapping speech, through python (*sPyeech*) to EEG (*sPyEEG*)... and the other way around! 

Not *mind-reading* for espionage purposes ;).

### Setup

#### Requirements
Package builds on top on [MNE](https://mne.tools/stable/index.html) and relies on a similar set of dependencies and 3rd party packages listed in ```environment.yml```. You can easily set up the environment via [Conda](https://docs.conda.io/en/latest/) package manager by running in terminal From terminal (or _conda shell_ on Windows): 
```bash
conda env update --file environment.yml
```
Then activate the created environment by running:
```bash
conda activate spyeeg
```

#### Installation
To get the package installed only through symbolic links, namely so that you can modify the source code and use modified versions at will when importing the package in your python scripts do:

```bash
python setup.py develop
```

Otherwise, for a standard installation (but this will require to be installed if you need to install another version of the library):

```bash
python setup.py install
```

##### Tested on:
- macOS Big Sur v11.1
- Ubuntu 18.04.5 LTS

### Modules (sketch)
- **models** - for all your modelling needs...
  - submodules tbd...
- **feat** - feature extraction...
- **preproc** - preprocessing...
- **viz** - visualizations...
- **utils** - misc.

### Examples
**Note**: Sample data required for demos can be downloaded [here](https://imperialcollegelondon.box.com/s/afalp7tysg6nlayb5hftyn5xopv6uh99). When downloaded place the files in the ```demos/Data``` folder.
- **Modelling** (```models```)
  - Basic TRF modelling: ```demo/Demo_TRF.py```
- **Feature extraction** (```feat```)
  - Speech envelope extraction: ```demo/Demo_envelopes.py```
- coming soon...

#### Contributors:
- Pierre Guilleminot (phg17@ic.ac.uk)
- Mikolaj Kegler (mak616@ic.ac.uk)

Last updated: 12th Jan 2021