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