#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 14:31:08 2022

@author: lau
"""

#%% IMPORTS OF PACKAGES

import mne
from os.path import join
import matplotlib.pyplot as plt
import pathlib

plt.ion() #toggle interactive plotting
# import numpy as np

#%% PATHS FOR EXAMPLE ANALYSIS
path = pathlib.Path(__file__)
meg_path = path.parents[3] / "834761" / "0114" / "20230927_000000" / "MEG" / "001.self_block1" / "files"

bem_path = path.parents[3] / "835482" / "0114" / "bem"

subjects_dir = path.parents[3] / "835482" 

raw_name = 'self_block1.fif'
fwd_name = 'self_block1-oct-6-src-5120-fwd.fif'

#%% READ RAW AND PLOT

raw = mne.io.read_raw_fif(join(meg_path, raw_name), preload=True)
raw.plot() ## what happens after 10 seconds?
raw.compute_psd(n_jobs=-1).plot()
raw.compute_psd(n_jobs=-1, tmax=9).plot()

#%% FILTER RAW

raw.filter(l_freq=None, h_freq=40, n_jobs=4) # alters raw in-place
raw.compute_psd(n_jobs=-1).plot()
raw.plot()

#%% CONCATENATION OF RAWS - cannot be done without MaxFilter
# and setting the destination argument

# mne.preprocess.maxwell_filter

#%% FIND EVENTS

# events = mne.find_events(raw)#, min_duration=0.002) ## returns a numpy array
events = mne.find_events(raw, min_duration=0.002) ## returns a numpy array

mne.viz.plot_events(events) ## samples on x-axis
mne.viz.plot_events(events, sfreq=raw.info['sfreq']) ## 

#%% SEGMENT DATA INTO EPOCHS

event_id = dict(self_positive=11, self_negative=12, button_press=23,
                incorrect_response=202)
# reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6) # T, T/m, V
reject = None
epochs = mne.Epochs(raw, events, event_id, tmin=-0.200, tmax=1.000,
                    baseline=(None, 0), reject=reject, preload=True,
                    proj=False) ## have proj True, if you wanna reject

epochs.pick_types(meg=True, eog=False, ias=False, emg=False, misc=False,
                  stim=False, syst=False)

epochs.plot()

#%% EVOKED - AVERAGE - projs not applied

evokeds = list()
for event in event_id:

    evoked = epochs[event].average()
    evokeds.append(evoked)
    evoked.plot(window_title=evoked.comment)
#%% PROJS
mne.viz.plot_projs_topomap(evoked.info['projs'], evoked.info)

epochs.apply_proj()

evokeds = list()
for event in event_id:

    evoked = epochs[event].average()
    evokeds.append(evoked)
    evoked.plot(window_title=evoked.comment)


#%% ARRAY OF INTEREST FOR CLASSIFICATIION

X = epochs.get_data()

#%% SOURCE RECONSTRUCTION

#%% read forward solution
fwd = mne.read_forward_solution(bem_path / fwd_name)
src = fwd['src'] # where are the sources
trans = fwd['mri_head_t'] # what's the transformation between mri and head
info = epochs.info # where are the sensors?
bem_sol = fwd['sol'] # how do electric fields spread from the sources inside the head?

bem = bem_path / "0114-5120-bem.fif"


#%% plot source space
src.plot(trans=trans, subjects_dir=str(subjects_dir))
src.plot(trans=fwd['mri_head_t'], subjects_dir=str(subjects_dir), head=True,
         skull='inner_skull')

mne.viz.plot_alignment(info, trans=trans, subject='0114',
                       subjects_dir=subjects_dir, src=src,
                       bem=bem, dig=True, mri_fiducials=True)

#%% estimate covariance in the baseline to whiten magnetometers and 
#   gradiometers
noise_cov = mne.compute_covariance(epochs, tmax=0.000)
noise_cov.plot(epochs.info) # not full range due to projectors projected out

#%% operator that specifies hpw noise cov should be applied to the fwd
evoked = evokeds[0]
inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, noise_cov)

#%% estimate source time courses for evoked responses
stc = mne.minimum_norm.apply_inverse(evoked, inv, method='MNE')
print(stc.data.shape)
print(src)

stc.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170)

#%% load the morph to the template brain - allows for averaging across subjects
morph_name = '0108-oct-6-src-morph.h5'
morph = mne.read_source_morph(join(bem_path, morph_name))

#%%# apply the morph to the subject - bringing them into template space
## this allows for averaging of subjects in source space
stc_morph = morph.apply(stc)
stc_morph.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170,
               subject='fsaverage')


#%%# reconstruct individual epochs instead of evoked
stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv,
                                             lambda2=1, method='MNE',
                                             pick_ori='normal')

# Mean across epochs - why do we have negative values now as well?
mean_stc = sum(stcs) / len(stcs)
mean_stc.plot(subjects_dir=subjects_dir, hemi='both', initial_time=0.170)



#%% reconstructing single labels

def reconstruct_label(label_name):
    label = mne.read_label(join(bem_path, '..', 'label', label_name))

    stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv,
                                             lambda2=1, method='MNE',
                                             pick_ori='normal', label=label)

    mean_stc = sum(stcs) / len(stcs) # over trials, not vertices
    return mean_stc

ltc = reconstruct_label('lh.BA44_exvivo.label')
## check the label path for more labels

plt.figure()
plt.plot(ltc.times, ltc.data.T)
plt.show()
