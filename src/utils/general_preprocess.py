import pathlib
import mne

def ica_dict():
    ica_dict = {
        "001.self_block1":[1, 5, 8], 
        "002.other_block1":[1, 7, 8], 
        "003.self_block2":[1, 5, 12], 
        "004.other_block2":[1, 8, 11], 
        "005.self_block3":[1, 7, 12], 
        "006.other_block3":[1, 9, 15]
        }

    return ica_dict

def preprocess(meg_path, recording_name, ica_path, ica_exclude:list):
    '''
    Preprocesses raw data for a single recording

    Args:
        meg_path (pathlib.Path): path to meg data
        recording_name (str): recording name
        ica_path (pathlib.Path): path to ICA data
        ica_exclude (list): list of ICA components to exclude

    Returns:
        processed_raw (mne.io.Raw): preprocessed raw data (where ica has been applied)
    '''

    # load raw
    fif_fname = recording_name[4:]
    full_path = meg_path / recording_name / 'files' / (fif_fname + '.fif')
    
    # read, load raw, pick types
    raw = mne.io.read_raw(full_path, preload=True)
    raw.load_data()
    raw.pick_types(meg=True, eog=False, stim=True)

    # remove bad channel
    raw.info['bads'] += ['MEG0422']
    raw.drop_channels(raw.info['bads'])

    # crop to remove initial HPI noise and noise at the end of each trial (verified by manually checking raws in run_raw.py)
    cropped = raw.copy().crop(tmin=10, tmax=365)
    del raw

    # initial filtering (back to 0.1 hz instead of 1 hz)
    filtered = cropped.copy().filter(l_freq=0.1, h_freq=40)
    filtered.apply_proj()

    # RESAMPLE 
    resampled = filtered.copy().resample(250)
    del filtered
    
    # load ICA 
    ica_full_path = ica_path / f"{recording_name}-ica.fif"
    ica = mne.preprocessing.read_ica(ica_full_path)

    # exclude icas 
    ica.exclude = ica_exclude

    # apply ICA
    processed_raw = resampled.copy()
    del resampled
    ica.apply(processed_raw)

    return processed_raw

def preprocess_all(meg_path, recording_names, ica_path, ica_dict):
    '''
    Preprocesses all recordings in recording_names

    Args
        meg_path (pathlib.Path): path to meg data
        recording_names (list): list of recording names
        ica_path (pathlib.Path): path to ICA data
        ica_dict (dict): dictionary of ICA exclude lists

    Returns:
        processed_raws (dict): dictionary of preprocessed raws (where ica has been applied)
    '''

    processed_raws = {}
    for _, name in enumerate(recording_names):
        ica_exclude = ica_dict[name]
        processed_raws[name] = preprocess(meg_path, name, ica_path, ica_exclude)

    return processed_raws

def epoching(raw, events, tmin, tmax, event_id=dict, reject_criterion:dict=None):
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

def create_evoked(epochs, triggers:list):
    '''
    Create evoked for the specified triggers
    '''
    evoked = {}

    for trigger in triggers: 
        # select epochs for specific trigger
        trigger_epochs = epochs[trigger]

        # average epochs for specific trigger
        evoked[trigger] = trigger_epochs.average()

    return evoked 