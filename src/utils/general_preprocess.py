import pathlib
import mne

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

def combine_raws(meg_path:pathlib.Path, recording_names:list, tmin=-0.200, tmax=1.500, reject_criterion=None): 
    '''
    Combines raw data from multiple recordings, filters (high pass of 40 hz) and epochs data. 

    Args
        meg_path (pathlib.Path): path to meg data (meg_path = .. / "834761" / "0114" / "20230927_000000" / "MEG" / recording_name)
        recording_names (list): names of the recording files (e.g., "001.self_block1", "002.other_block1", etc.)
        reject_criterion (dict): dictionary of reject criterion for epoching

    Returns 
        epochs (mne.Epochs): epoched data for all recordings.
    '''
    
    epochs_list = []
    
    # iterate over recording names
    for recording_index, recording_name in enumerate(recording_names):
        # define file names, paths 
        fif_fname = recording_name[4:]
        full_path = meg_path / recording_name / 'files' / (fif_fname + '.fif')
        
        # read raw and filter 
        raw = mne.io.read_raw(full_path, preload=True)
        raw.filter(l_freq=None, h_freq=40, n_jobs=3)
        
        # find events 
        events = mne.find_events(raw, min_duration=0.002)

        # combining all triggers to 
        if 'self' in recording_name:
            event_id = dict(positive=11, negative=12,
                            button_img=23)
        
        elif 'other' in recording_name: 
            event_id = dict(positive=21, negative=22,
                            button_img=23)
        else:
            raise NameError('Event codes are not coded for file')
        
        # added a epoching function which includes a reject criterion that was not present in OG script
        epochs = epoching(raw, events, tmin=tmin, tmax=tmax,
                          event_id=event_id, reject_criterion=reject_criterion)
        
        epochs_list.append(epochs)

    return epochs_list

def create_evoked(epochs_list, triggers:list):
    '''
    Create evoked for the specified triggers
    '''
    evoked = {}

    for trigger in triggers: 
        # select epochs for specific trigger
        trigger_epochs = epochs_list[trigger]

        # average epochs for specific trigger
        evoked[trigger] = trigger_epochs.average()

    return evoked 