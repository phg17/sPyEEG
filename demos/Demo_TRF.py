"""
Created on Wed Dec 16 18:21:39 2020

@author: phg17
"""

import numpy as np
from spyeeg.models.TRF import TRFEstimator
from os.path import join
import mne

path_data = './Data'
Fs = 256


def get_raw_info():
    fname = join(path_data, 'info')
    raw = mne.io.read_raw_fif(fname, preload=True)
    raw.set_eeg_reference('average', projection=False)
    raw.drop_channels(['Sound', 'Diode', 'Button'])
    return raw.info

#Get Topology Info
info = get_raw_info()
info['sfreq'] = Fs

#Load Data
envelope = np.load(join(path_data, 'envelope_1.npy'))
eeg = np.load(join(path_data, 'eeg_1.npy'))
vowels = np.load(join(path_data, 'vowels_1.npy'))

#Create Features matrix 
xtrf = np.hstack([envelope, vowels])
ytrf = eeg[:]  # Make a deep copy?

#Fit
trf = TRFEstimator(tmin=-.5, tmax=.5, srate=Fs, alpha=[10])
trf.fit_from_cov(xtrf, ytrf, part_length=60, clear_after=False)

#Extract Coef
coef_envelope = trf.get_coef()[:, 0, :, 0].T
coef_vowels = trf.get_coef()[:, 1, :, 0].T

#Plot using mne 
ev1 = mne.EvokedArray(coef_envelope, info, tmin=trf.tmin)
ev2 = mne.EvokedArray(coef_vowels, info, tmin=trf.tmin)

ev1.plot_joint()
ev2.plot_joint()
