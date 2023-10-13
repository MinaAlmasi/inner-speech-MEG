'''
Sanity check for visual activation.
'''
# utils 
import pathlib 
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

import mne

# custom module
from src.utils.general_preprocess import preprocess, epoching, ica_dict

def main(): 
    # define paths 
    path = pathlib.Path(__file__)

    meg_path = path.parents[4] / "834761" / "0108" / "20230928_000000" / "MEG"
    ica_path = path.parents[2] / "data" / "ICA"

    plots_path = path.parents[1] / "plots" / "sanity_checks"
    plots_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

    # define recording names 
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']

    # define recording name
    chosen_recording = recording_names[1]

    # print ica components to exclude
    ica_components = ica_dict()
    
    # load and preprocess data
    processed_raw = preprocess(meg_path, chosen_recording, ica_path, ica_components[chosen_recording])

    # get events
    events  = mne.find_events(processed_raw, min_duration = 2/processed_raw.info["sfreq"])

    ## EPOCHING ## 
    
    # check if self or other block to determine event ids
    if "self" in chosen_recording: 
        event_id = dict(self_positive=11, self_negative=12, button_img=23)
    else: 
        event_id = dict(other_positive=21, other_negative=22, button_img=23)

    # do epochs 
    reject = dict(mag=4e-12, grad=4000e-13) # T, T/m, V

    epochs = epoching(processed_raw, events, event_id=event_id, tmin=-0.200, tmax=1.000, reject_criterion=reject)
    print(epochs)

    plot = epochs.plot_image(picks="meg", combine="mean")

    # resample to 250 Hz
    epochs = epochs.resample(250)

    # save plot
    plot.savefig(plots_path / "epochs.png")

    # compute evokeds
    evoked = create_evoked(epochs, triggers=['other_positive', 'other_negative', 'button_img'])

    pos_evoked =  evoked['other_positive']
    neg_evoked = evoked['other_negative']
    button_evoked = evoked['button_img']

    # combine evoked 
    visual_evoked = mne.combine_evoked([pos_evoked, neg_evoked, button_evoked], weights="equal")
    
    # plot evokeds
    pos_evoked_plot = visual_evoked.plot_joint(picks="meg")
    pos_evoked_plot[1].savefig(plots_path / f"all_evoked_plot_{chosen_recording[4:]}.png", dpi=1200)

    # plot topographies
    #pos_evoked_topo_plot = visual_evoked.plot_topomap(times=[-0.1, 0.0, 0.3, 0.8, 1.4], ch_type="grad")
    #pos_evoked_topo_plot.savefig(plots_path / "pos_evoked_topo_plot.png")


if __name__ == "__main__":
    main()