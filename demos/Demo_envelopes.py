"""
Demo showing how to use the envelope extration from spyeeg.feat module 

@author: mak616
"""

import spyeeg
import scipy.io.wavfile as wav
import sys
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Load some sample audio data
fs_audio, audio = wav.read('./Data/audio_sample.wav')
fs_ds = 100  # Downsampling frequency in Hz

# Choosing cutoff frequencies for filtering
# Scalar value indicates corner frequency of the low pass filter
env_lp = spyeeg.feat.signal_envelope(audio, fs_audio, cutoff=8, resample=fs_ds)

# List-like object with 2 values - corner frequencies of bandpass filter
env_bp = spyeeg.feat.signal_envelope(
    audio, fs_audio, cutoff=[4, 8], resample=fs_ds)

# Any other combination will lead to an error
try:
    env_bad = spyeeg.feat.signal_envelope(
        audio, fs_audio, cutoff=[1, 2, 3, 5], resample=fs_ds)
except ValueError as err:
    print(err)

# Function accepts remaining keyword args from the mne.filter.create_filter
# (https://mne.tools/dev/generated/mne.filter.create_filter.html) for more flexibility
# In this example we enable verbosity (handy) and tune transition bands
env_custom = spyeeg.feat.signal_envelope(audio, fs_audio, cutoff=[4, 8], resample=fs_ds, \
                                         verbose=True, l_trans_bandwidth=1, h_trans_bandwidth=1)

# Plotting
f, ax = plt.subplots(1, 3, sharex=True, figsize=(15, 5))
time = np.arange(300)/fs_ds
ax[0].plot(time, env_lp[300:600])
ax[1].plot(time, env_bp[300:600])
ax[2].plot(time, env_custom[300:600])
for a in ax:
    a.set_xlabel('Time (s)')
    a.set_ylabel('Amplitude (a.u.)')
ax[0].set_title('8 Hz lowpass')
ax[1].set_title('4-8 Hz bandpass')
ax[2].set_title('4-8 Hz bandpass (custom - sharp)')
f.tight_layout()
plt.show()
# f.savefig('Envs.pdf')
