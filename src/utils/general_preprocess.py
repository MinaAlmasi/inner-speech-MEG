import pathlib
import mne

def combine_raws(meg_path:pathlib.Path, recording_names:list): 
    '''
    Combines raw data from multiple recordings.

    Args
        meg_path (pathlib.Path): path to meg data (meg_path = .. / "834761" / "0114" / "20230927_000000" / "MEG" / recording_name)
        recording_names (list): names of the recording files (e.g., "001.self_block1", "002.other_block1", etc.)

    Returns 
        raw (mne.io.Raw): raw data from all recordings
    '''
    
    raw_files = []
    
    # iterate over recording names
    for recording_index, recording_name in enumerate(recording_names):
        # define file names, paths 
        fif_fname = recording_name[4:]
        full_path = meg_path / recording_name / 'files' / (fif_fname + '.fif')

        # read raw and MAXWELL FILTER 
        raw = mne.io.read_raw(full_path, preload=True)

        # set dev_head_t_ref
        if recording_index == 0:
            dev_head_t_ref = raw.info['dev_head_t']
        
        raw = mne.preprocessing.maxwell_filter(raw, origin='auto', coord_frame='head', destination=dev_head_t_ref)           

        raw_files.append(raw)

    # combine raws
    combined_raw = mne.concatenate_raws(raw_files, preload=True)

    return combined_raw

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

def get_events(raw, min_duration=0.002): 
    '''
    Gets events from raw data.
    Args:
        raw (mne.io.Raw): raw data

    Returns:
        events (numpy array): events
    '''
    events = mne.find_events(raw, min_duration=min_duration) ## returns a numpy array

    return events

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