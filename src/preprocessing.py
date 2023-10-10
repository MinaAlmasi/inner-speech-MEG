import pathlib
import numpy as np
import mne
from os.path import join
import matplotlib.pyplot as plt

import argparse

def load_raw(meg_path, raw_name):
    raw_name = meg_path/raw_name
    raw = mne.io.read_raw_fif(fname=raw_name, preload=True)
    return raw

def filter_raw(raw, l_freq=None, h_freq=40):
    '''
    Apply low or high pass filtering to raw data.
    Args:
        raw (mne.io.Raw): raw data
        l_freq (float): low frequency
        h_freq (float): high frequency

    Returns:
        raw (mne.io.Raw): filtered raw data
    '''
    raw.filter(l_freq=l_freq, h_freq=h_freq, n_jobs=4)

    return raw 

def get_events(raw): 
    '''
    Gets events from raw data.
    Args:
        raw (mne.io.Raw): raw data

    Returns:
        events (numpy array): events
    '''
    events = mne.find_events(raw, min_duration=0.002) ## returns a numpy array

    return events

def epoching(raw, events, tmin=-0.200, tmax=1.000, event_id=dict, reject_criterion:dict=None):
    '''
    Epochs raw data based on events and event_id. Rejects epochs based on reject_criterion if specified.

    Args:
        raw (mne.io.Raw): raw data
        events (numpy array): events
        tmin (float): start of epoch in seconds
        tmax (float): end of epoch in seconds
        event_id (dict): dictionary of events
        reject_criterion (dict): dictionary of reject criterion

    Returns:
        epochs (mne.Epochs): epoched data
    '''

    if reject_criterion:
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=(None, 0), reject=reject_criterion, preload=True,
                    proj=True) # have proj = True if you want to reject 

    else: 
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                    baseline=(None, 0), reject=reject_criterion, preload=True,
                    proj=False)
    
    epochs.pick_types(meg=True, eog=False, ias=False, emg=False, misc=False,
                        stim=False, syst=False)

    # apply projections
    epochs.apply_proj()

    return epochs 

def compute_evoked(epochs, event_id:dict):
    '''
    Computes evoked data based on epochs and event_id.

    Args:
        epochs (mne.Epochs): epoched data
        event_id (dict): dictionary of events

    Returns:
        evokeds (list): list of evoked data
    '''
    evokeds = []

    for event in event_id:
        evoked = epochs[event].average()
        evokeds.append(evoked)

    return evokeds 

def input_parse(): 
    parser=argparse.ArgumentParser(description='Preprocess MEG data')

    # add number of block as arg
    parser.add_argument('--block', type=int, help='block number')

    # add condition as arg
    parser.add_argument('--condition', type=str, help='condition')

def get_block(number, condition):
    '''fix later'''
    return f"00{number}.{condition}_block{number}"

def main(): 
    # define paths 
    path = pathlib.Path(__file__)

    meg_path = path.parents[3] / "834761" / "0114" / "20230927_000000" / "MEG" / "001.self_block1" / "files"

    bem_path = path.parents[3] / "835482" / "0114" / "bem"

    subjects_dir = path.parents[3] / "835482" 

    raw_name = 'self_block1.fif'
    fwd_name = 'self_block1-oct-6-src-5120-fwd.fif'

    # load raw data
    raw = load_raw(meg_path, raw_name)
    
    # filter raw data
    raw = filter_raw(raw)

    # get events
    events = get_events(raw)

    # event dict
    #event_dict = {"IMG_POS":11, "IMG_NEG":12, "IMG_BUTTON":23, "BUTTON_PRESS":202, "MISCLICK":103}
    event_id = dict(img_positive=11, img_negative=12, img_button=23,
                button_press=202)

    # reject criterion
    reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6) # T, T/m, V

    # epoching
    epochs = epoching(raw, events, event_id=event_id, reject_criterion=reject)

    # compute evoked
    evokeds = compute_evoked(epochs, event_id)

if __name__ == "__main__":
    main()

