"""
ERP-style analysis.
Fundamental Q - are we doing it better/more efficiently than MNE itself? (i.e. reinventing the wheel)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import lag_span, lag_sparse, get_timing
import mne


class ERP_class():
    def __init__(self, tmin, tmax, srate, n_chan=63):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.window = lag_span(tmin, tmax, srate)
        self.times = self.window/srate
        self.ERP = np.zeros(len(self.window))
        self.mERP = np.zeros([len(self.window), n_chan])
        self.single_events = []
        self.single_events_mult = []
        self.peak_time = None
        self.peak_arg = None

    def add_data(self, eeg, events, event_type='spikes'):

        if event_type == 'spikes':
            events_list = get_timing(events)
        else:
            events_list = events

        # for i in np.where(events_list < eeg.shape[0] - self.window[-1])[0]:
        for i in range(len(events_list)):
            try:
                event = events_list[i]
                self.ERP += np.sum((eeg[self.window + event]), axis=1)
                self.mERP += eeg[self.window + event, :]
                self.single_events.append(
                    np.sum((eeg[self.window + event]), axis=1))
            except:
                print('out of window')
        self.peak_time = np.argmax(self.ERP) / self.srate + self.tmin
        self.peak_arg = np.argmax(self.ERP) + self.tmin * self.srate

    def weight_data(self, eeg, cont_stim):

        # for i in np.where(events_list < eeg.shape[0] - self.window[-1])[0]:
        for i in range(len(cont_stim)):
            try:
                w = cont_stim[i]
                self.ERP += np.sum((w * eeg[self.window + i]), axis=1)
                self.mERP += w * eeg[self.window + i, :]
                self.single_events.append(
                    np.sum(np.abs(w * eeg[self.window + i]), axis=1))
                self.single_events_mult.append((w * eeg[self.window + i]))
            except:
                pass

    def inverse_weight_data(self, eeg, cont_stim):

        for n_chan in range(63):
            for t in range(len(cont_stim)):
                try:
                    w = eeg[t, n_chan]
                    self.mERP[:, n_chan] += w * cont_stim[self.window + t, 0]
                except:
                    pass
        self.ERP = np.sum(self.mERP, axis=1)

    def plot_simple(self):
        plt.figure()
        plt.plot(self.times, self.ERP)

    def plot_multi(self):
        plt.figure()
        plt.plot(self.times, self.mERP)

    def plot_topo(self, raw_info, Fs, time=None):

        f, (ax1, ax2) = plt.subplots(1, 2)
        f.set_figwidth(10)
        if not time:
            t = [np.argmin(self.ERP), np.argmax(self.ERP)]
        else:
            t = [int((time[0] - self.tmin)*Fs), int((time[1] - self.tmin)*Fs)]

        # Visualize topography max
        mne.viz.plot_topomap(
            self.mERP[t[0], :], raw_info, axes=ax1, cmap='RdBu_r', show=False)
        ax1.set_title('tlag={} ms'.format(
            int((t[0]/self.srate + self.tmin)*1000)))

        # Visualize topography min
        mne.viz.plot_topomap(
            self.mERP[t[1], :], raw_info, axes=ax2, cmap='RdBu_r', show=False)
        ax2.set_title('tlag={} ms'.format(
            int((t[1]/self.srate + self.tmin)*1000)))
