"""
ERP-style analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ..utils import lag_span, lag_sparse, get_timing
import mne
from matplotlib import colormaps as cmaps


class ERP_class():
    """
    This class mostly helps handling data to manipulate it in ERP forms. It is a simplified version of mne.Epochs object and does not depend on the mne environment, but rather provide basic array manipulation.

    Parameters
    ----------
    tmin : float
        Minimum timelag, in seconds
    tmax : float
        Maximum timelag, in seconds
    srate : float
        Sampling rate
    baseline : tuple
        tuple, with the time to consider to compute the baseline. If None, no baseline correction is applied. 
    """
    
    def __init__(self, tmin, tmax, srate, baseline = None):
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
        self.baseline = baseline
        if not (baseline is None):
            self.baseline_window = lag_span(baseline[0] - tmin, baseline[1] - tmin, srate)
        else:
            self.baseline_window = ()
        

    def add_events(self, eeg, events, weights = None,
                   event_type = 'feature', 
                   weight_events = False):
        """
        Compute ERPs object based on discrete events i.e. values are taken at specific time in the neural signal.

        Parameters
        ----------
        eeg : ndarray
            eeg data, of shape (T, nchan)
        events : ndarray 
            Either continuous signal, in each case non-zero value will be treated as events, or array-like of indices, representing the onset of events
        weights : ndarray
            If events are indices, these are the corresponding weights of events. If None, they are set to 1.
        event_type : str
            Either "feature" (continuous input) or "spikes" (events indices)
        weight_events : bool
            Whether to weight neural data according to the events. 
        """

        self.n_chans_ = eeg.shape[1]
        self.mERP = np.zeros([len(self.window), self.n_chans_])
        self.evoked = []
        self.weights = []
        self.events = []

        if event_type == 'feature':
            events, weights = get_timing(events)
        elif event_type == 'spikes' and not (weights is None):
            assert len(events) == len(weights), "events and weights must have equal length to use spikes"
        elif event_type == 'spikes' and weights is None:
            weights = np.ones(len(events))

        for i in range(len(events)):
                event = int(events[i])
                weight = weights[i]

                if event + self.window[-1] < eeg.shape[0]:
                    if len(self.baseline_window) > 0:
                        baseline_correction = eeg[self.baseline_window,:].mean(0)
                    else: 
                        baseline_correction = 0
                        
                    if weight_events:
                        data = (eeg[self.window + event, :] - baseline_correction) * weight
                    else:
                        data = eeg[self.window + event, :] - baseline_correction
                    self.mERP += data
                    self.evoked.append(data)
                    self.events.append(event)
                    self.weights.append(weight)
                    
        self.mERP /= len(self.events)

    

    def add_continuous_signal(self, eeg, signal, step = None, weight_events = False, record_weight = True):
        if step is None:
            step = len(self.window)
        self.n_chans_ = eeg.shape[1]
        self.mERP = np.zeros([len(self.window), self.n_chans_])
        self.evoked = []
        self.weights = []
        self.events = []

        event = 0
        while event + self.window[-1] < eeg.shape[0]:
            if weight_events or record_weight:
                weight = signal[event]
            else:
                weight = 1
            if weight_events:
                data = weight * eeg[self.window + event, :]
            else:
                data = eeg[self.window + event, :]
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

