# utils 
import pathlib 

import mne

# custom module
from utils.general_preprocess import combine_raws, create_evoked, filter_raw, get_events, epoching

def plot_visual_topography():
    pass

def main(): 
    # define paths 
    path = pathlib.Path(__file__)

    meg_path = path.parents[3] / "834761" / "0114" / "20230927_000000" / "MEG"

    plots_path = path.parents[1] / "plots" / "sanity_checks"
    plots_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

    # define recording names 
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']
    
    # combine raws
    #raw = combine_raws(meg_path, recording_names)

    # load raw
    raw = mne.io.read_raw(meg_path / recording_names[1] / 'files' / 'other_block1.fif', preload=True)

    # filter raw
    filtered_raw = filter_raw(raw)

    # get events
    events = get_events(filtered_raw)
    print(events)

    # segment data into epochs
    #event_id = dict(self_positive=11, other_positive=21, self_negative=12, other_negative=22, button_img=23)
    #event_id = dict(self_positive=11, self_negative=12, button_img=23)
    event_id = dict(other_positive=21, other_negative=22, button_img=23)
    reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6) # T, T/m, V
    
    epochs = epoching(filtered_raw, events, event_id=event_id, tmin=-0.200, tmax=1.000, reject_criterion=reject)

    # combine event ids 
    #epochs = mne.epochs.combine_event_ids(epochs, ['self_positive', 'other_positive'], {'positive': 1121})
    #epochs = mne.epochs.combine_event_ids(epochs, ['self_negative', 'other_negative'], {'negative': 1222})

    print(epochs)
    plot = epochs.plot_image(picks="meg", combine="mean")

    # resample to 250 Hz
    epochs = epochs.resample(250)

    # save plot
    #plot.savefig(plots_path / "epochs.png")

    # compute evoked 
    evoked = create_evoked(epochs, triggers=['other_positive', 'other_negative', 'button_img'])

    pos_evoked =  evoked['other_positive']
    neg_evoked = evoked['other_negative']
    button_evoked = evoked['button_img']

    # combine evoked 
    visual_evoked = mne.combine_evoked([pos_evoked, neg_evoked, button_evoked], weights="equal")
    
    # plot evokeds
    pos_evoked_plot = visual_evoked.plot_joint(picks="meg")
    pos_evoked_plot[1].savefig(plots_path / "all_evoked_plot_otherblock1.png", dpi=1200)

    # plot topographies
    #pos_evoked_topo_plot = visual_evoked.plot_topomap(times=[-0.1, 0.0, 0.3, 0.8, 1.4], ch_type="grad")
    #pos_evoked_topo_plot.savefig(plots_path / "pos_evoked_topo_plot.png")


if __name__ == "__main__":
    main()