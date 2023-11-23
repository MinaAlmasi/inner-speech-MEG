'''
Plot to create a contrast between self and other conditions
'''
import numpy as np
import mne
import pathlib

from utils.general_preprocess import preprocess_all, ica_dict, epoching
from utils.classify_fns import combine_triggers

def get_source_time_courses(epochs_dict:dict, subjects_dir, subject:str="0108", label=None, method="dSPM"):
    '''
    Extract source space data for contrasts 
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
    
    combined_stcs = []

    for epochs_index, (recording_name, epochs) in enumerate(epochs_dict.items()):
        fwd_name = f"{recording_name[4:]}-oct-6-src-5120-fwd.fif"

        # read forward solution
        fwd = mne.read_forward_solution(subjects_dir / subject / 'bem' / fwd_name)

        # source estimation! 
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1,
                                                     method=method, label=label,
                                                     pick_ori="normal")

        # concatenate with previous stcs
        combined_stcs.extend(stcs)

    return combined_stcs, y 

def split_stcs(stcs_list, y, trigger1, trigger2):
    '''
    Split stcs into self and other conditions
    '''
    # get indices for self and other conditions
    indices_trigger1 = np.where(y == trigger1)[0]
    indices_trigger2 = np.where(y == trigger2)[0]

    # split stcs into self and other conditions
    stcs_trigger1 = [stcs_list[i] for i in indices_trigger1]
    stcs_trigger2 = [stcs_list[i] for i in indices_trigger2]

    return stcs_trigger1, stcs_trigger2

def plot_stcs(stcs, subjects_dir, subject="0108", savepath=None):
    '''
    Plot source time course object 
    '''
    # get mean stc
    mean_stcs = np.mean(stcs)

    # plot params
    clim = dict(kind='value', lims=[0.35, 0.85, 1.4])
    plot = mean_stcs.plot(
        initial_time=0.30,
        subjects_dir=subjects_dir,
        subject=subject,
        hemi='split',
        size = (800, 400),
        views = ["lateral"],
        clim=clim
    )

    if savepath is not None:
        plot.save_image(savepath)
    
    return plot

def main(): 
    ## PATHS and FILES ## 
    path = pathlib.Path(__file__)

    # raw meg data paths 
    meg_path = path.parents[3] / "834761" / "0108" / "20230928_000000" / "MEG"
    ica_path = path.parents[1] / "data" / "ICA"
    subjects_dir = path.parents[3] / "835482" 

    # plot path
    plot_path = path.parents[1] / "plots" / "stc_plots"
    plot_path.mkdir(parents=True, exist_ok=True)

    ## LOAD + PREPROCESS DATA ##
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']

    # get ica components to exclude
    ica_components = ica_dict()

    # preprocess all recordings
    processed_raws = preprocess_all(meg_path, recording_names, ica_path, ica_components)

    # prepare for epochs, define rejection criterion
    epochs_dict = {}
    reject_criterion = dict(mag=4e-12, grad=4000e-13)

    # iterate over values in processed_raws
    for recording_name, raw in processed_raws.items(): 
        if "self" in recording_name: 
            event_id = dict(self_positive=11, self_negative=12, button_img=23)
        else: 
            event_id = dict(other_positive=21, other_negative=22, button_img=23)

        # get events
        events  = mne.find_events(raw, min_duration = 2/raw.info["sfreq"])

        # epoch data
        epochs = epoching(raw, events, tmin=-0.200, tmax=1.500, event_id=event_id, reject_criterion=reject_criterion)

        # append to dict
        epochs_dict[recording_name] = epochs

    # get source space data
    stcs, y = get_source_time_courses(epochs_dict, subjects_dir, subject="0108", label=None)

    # combine triggers
    y = combine_triggers(y, combine=[[11, 21], [12, 22]])

    # get stcs for the two groups (based on triggers)
    stcs_1, stcs_2 = split_stcs(stcs, y, trigger1=1121, trigger2=1222)

    # plot contrast for both stcs and stcs2 using plot_stcs:
    plot_stcs(stcs_1, subjects_dir, subject="0108", savepath=plot_path / "positive_self_and_other.png")
    plot_stcs(stcs_2, subjects_dir, subject="0108", savepath=plot_path / "negative_self_and_other.png")

if __name__ == "__main__": 
    main()
