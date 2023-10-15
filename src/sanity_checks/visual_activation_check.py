'''
Sanity check for visual activation.

Run in terminal: 
    python src.sanity_checks.visual_activation_check -r 0
'''
# utils 
import pathlib 
import sys
sys.path.append(str(pathlib.Path(__file__).parents[2]))

import mne

# custom module
from src.utils.general_preprocess import preprocess, epoching, ica_dict, create_evoked
from src.utils.arguments import input_parse

def main(): 
    # args
    args = input_parse()

    # define paths 
    path = pathlib.Path(__file__)

    meg_path = path.parents[4] / "834761" / "0108" / "20230928_000000" / "MEG"
    ica_path = path.parents[2] / "data" / "ICA"

    plots_path = path.parents[2] / "plots" / "sanity_checks"
    plots_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

    # define recording names 
    recording_names = {0: '001.self_block1',  1: '002.other_block1',
                       2: '003.self_block2',  3: '004.other_block2',
                       4: '005.self_block3',  5: '006.other_block3'}

    # define recording name
    chosen_recording = recording_names[args.recording]

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
        # define condition
        cond = "self"
    else: 
        event_id = dict(other_positive=21, other_negative=22, button_img=23)
        cond = "other"

    # do epochs 
    reject = dict(mag=4e-12, grad=4000e-13) # T, T/m, V

    epochs = epoching(processed_raw, events, event_id=event_id, tmin=-0.200, tmax=1.000, reject_criterion=reject)
    print(epochs)

    plots = epochs.plot_image(picks="meg", combine="mean")

    # save plot
    for i, _ in enumerate(plots): 
        plots[i].savefig(plots_path / f"{chosen_recording[4:]}_epochs_{i}.png")

    # compute evokeds
    evoked = create_evoked(epochs, triggers=[f'{cond}_positive', f'{cond}_negative', 'button_img'])

    pos_evoked =  evoked[f'{cond}_positive']
    neg_evoked = evoked[f'{cond}_negative']
    button_evoked = evoked['button_img']

    # combine evoked 
    visual_evoked = mne.combine_evoked([pos_evoked, neg_evoked, button_evoked], weights="equal")
    
    # plot evokeds
    pos_evoked_plot = visual_evoked.plot_joint(picks="meg")
    pos_evoked_plot[1].savefig(plots_path / f"all_evoked_plot_{chosen_recording[4:]}.png", dpi=1200)

    # plot topographies
    pos_evoked_topo_plot = visual_evoked.plot_topomap(times=[-0.2, 0.12, 0.5, 0.82, 1], ch_type="grad")
    pos_evoked_topo_plot.savefig(plots_path / f"pos_evoked_topo_plot_{chosen_recording[4:]}.png")


if __name__ == "__main__":
    main()