"""
ERP-style analysis.
Fundamental Q - are we doing it better/more efficiently than MNE itself? (i.e. reinventing the wheel)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import lag_span, lag_sparse, get_timing
import mne
from matplotlib import colormaps as cmaps


class ERP_class():
    def __init__(self, tmin, tmax, srate):
        self.srate = srate
        self.tmin = tmin
        self.tmax = tmax
        self.window = lag_span(tmin, tmax, srate)
        self.times = self.window/srate
        self.events = None
        self.weights = None
        self.evoked = None
        self.mERP = None
        self.n_chans_ = None

    def add_events(self, eeg, events, event_type='spikes', weight_events = False, record_weight = True, ignore_limit = False, scale_weights = True):

        self.n_chans_ = eeg.shape[1]
        self.mERP = np.zeros([len(self.window), self.n_chans_])
        self.evoked = []
        self.weights = []
        self.events = []

        events, weights = get_timing(events)
        if not weight_events and not record_weight :
            weights = np.ones(len(events))

        for i in range(len(events)):
                event = int(events[i])
                weight = weights[i]

                if event + self.window[-1] < eeg.shape[0]:
                    if weight_events:
                        data = eeg[self.window + event, :] * weight
                    else:
                        data = eeg[self.window + event, :] 
                    self.mERP += data
                    self.evoked.append(data)
                    self.events.append(event)
                    self.weights.append(weight)
                    
        self.mERP /= len(self.events)

    def add_continuous_signal(self, eeg, signal, step = None, weight_events = True):
        if step is None:
            step = len(self.window)
        self.n_chans_ = eeg.shape[1]
        self.mERP = np.zeros([len(self.window), self.n_chans_])
        self.evoked = []
        self.weights = []
        self.events = []

        event = 0
        while event + self.window[-1] < eeg.shape[0]:
            if weight_events:
                weight = signal[event]
            else:
                weight = 1
            data = weight * eeg[self.window + event, :]
            self.mERP += data
            self.evoked.append(data)
            self.events.append(event)
            self.weights.append(weight)
            event += step

        self.mERP /= len(self.events)


    

    def plot_ERP(self, figax = None, figsize = (10,5), color_type = 'jet', center_line = True,
                    channels = None, features = None, title = 'ERP'):
        """Plot the TRF of the feature requested as a *butterfly* plot"""
        if figax == None:
            fig,ax = plt.subplots(figsize = figsize, sharex = True)
        else:
            fig,ax = figax
        if channels == None:
            channels = np.arange(self.n_chans_)

        color_map = dict()
        for index_channel in range(self.n_chans_):
            color_map[index_channel] = cmaps[color_type](index_channel/self.n_chans_)


        for chan_index in range(self.n_chans_):
            chan = channels[chan_index]
            ax.plot(self.window/self.srate, self.mERP[:,chan_index], color = color_map[chan_index], linewidth = 1.5, label = chan)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('ERP (V)')
        if center_line:
            ax.plot([0,0],[np.min(self.mERP),np.max(self.mERP)], color = 'k', linewidth = 1.5, linestyle = '--')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1.15, 0.8),loc='right')
        ax.set_title(title)
        return fig,ax

