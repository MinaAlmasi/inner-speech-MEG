'''
Sanity check for motor vs visual activation (classification).

Run in terminal: 
    python src/sanity_checks/motor_visual_check.py
'''

# utils
import pathlib, sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

# MEG package
import mne

# numpy
import numpy as np

# custom modules for preprocessing and classification
from src.utils.general_preprocess import preprocess_all, ica_dict, epoching
from src.utils.classify_fns import simple_classification, plot_classification, get_source_space_data

def main(): 
    ## PATHS and FILES ## 
    path = pathlib.Path(__file__)

    # raw meg data paths 
    meg_path = path.parents[4] / "834761" / "0108" / "20230928_000000" / "MEG"
    ica_path = path.parents[2] / "data" / "ICA"
    subjects_dir = path.parents[4] / "835482" 

    # plot path
    plot_path = path.parents[2] / "plots" / "sanity_checks" / "motor_visual"
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
    label = "lh.postcentral.label"
    X, y = get_source_space_data(epochs_dict, subjects_dir, subject="0108", label=label)

    # get first value from epochs_dict
    first_epochs = list(epochs_dict.values())[0]
    times = first_epochs.times

    # select triggers for positive and button img
    triggers = [11, 21, 23]

    # complete simple classification
    classification = simple_classification(
                                X=X, 
                                y=y, 
                                triggers=triggers,
                                penalty='l2', 
                                C=1e-3, 
                                combine=[[11, 21]]) # combines the two positive triggers
    
    plot_classification(
        times = times, 
        mean_scores = classification, 
        title = f"{label}. Triggers: {triggers}",
        savepath = plot_path / f"{label}_{triggers}.png"
    )

if __name__ == "__main__":
    main()