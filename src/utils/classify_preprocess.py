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
    Extract and preprocess 

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
                            
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if stc_index == 0:
                n_trials = len(stcs)
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))
            
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))

    return X, y    

"""
def preprocess_source_space_data(subject, date, raw_path, subjects_dir,
                                 epochs_list, recording_names,
                              method='MNE', lambda2=1, pick_ori='normal',
                              label=None):

    # extract y 
    y = np.zeros(0)
    for epochs in epochs_list: # get y
        y = np.concatenate((y, epochs.events[:, 2]))
    
    if label is not None:
        label_path = join(subjects_dir, subject, 'label', label)
        label = mne.read_label(label_path)
    
    for epochs_index, epochs in enumerate(epochs_list): 
        # paths 
        recording = epochs
        f'{chosen_recording[4:]}-oct-6-src-5120-fwd.fif'

        fwd = mne.read_forward_solution(join(subjects_dir,
                                             subject, 'bem', fwd_fname))

        # source estimation! 
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2,
                                                     method, label,
                                                     pick_ori=pick_ori)
        for stc_index, stc in enumerate(stcs):
            this_data = stc.data
            if epochs_index == 0 and stc_index == 0:
                n_trials = len(stcs)
                n_vertices, n_samples = this_data.shape
                this_X = np.zeros(shape=(n_trials, n_vertices, n_samples))
            this_X[stc_index, :, :] = this_data
            
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))

    return X, y
    """