'''
Function for preparing data for classification
'''

# general utils
import pathlib
import numpy as np

# mne tools
import mne

def get_source_space_data(epochs_dict:dict, subjects_dir, subject:str="0108", label=None):
    '''
    Extract source space data for classification 
    (loosely based on https://mne.tools/stable/auto_examples/decoding/decoding_spatio_temporal_source.html#ex-dec-st-source)

    Args
        epochs_dict (dict): dictionary with epochs for each recording (keys are recording names, values are epochs objects)
        subjects_dir (str): path to subjects_dir
        subject (str): subject name (defaults to "0108")
        label (str): label name
    '''


    # set empty array for y
    y = np.zeros(0)

    # extract y for all epochs and concatenate
    for epochs in epochs_dict.values():
        y = np.concatenate((y, epochs.events[:, 2]))

    # load labels if relevant (if None, it will do a whole brain analysis)
    if label is not None:
        label_path = subjects_dir / subject / 'label' / label
        label = mne.read_label(label_path)
    
    for epochs_index, (recording_name, epochs) in enumerate(epochs_dict.items()):
        fwd_name = f"{recording_name[4:]}-oct-6-src-5120-fwd.fif"

        # read forward solution
        fwd = mne.read_forward_solution(subjects_dir / subject / 'bem' / fwd_name)

        # source estimation! 
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1,
                                                     method="MNE", label=label,
                                                     pick_ori="normal")

        # extract source space
        this_X = np.array([stc.data for stc in stcs])

        # concatenate (if first iteration, create X)
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    
    return X, y 