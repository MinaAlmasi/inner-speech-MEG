'''
Sanity check script to check the alignment of the sensors on the subject's head.
'''
# utils 
import pathlib, argparse, sys 
sys.path.append(str(pathlib.Path(__file__).parents[2]))

# MEG package
import mne

# custom modules for preprocessing
from src.utils.general_preprocess import preprocess, ica_dict, epoching
from src.utils.arguments import input_parse

# plotting 
import matplotlib.pyplot as plt

def save_3D_figure(plot, savepath): 
    '''
    Function inspired by https://mne.discourse.group/t/how-to-save-plot-sensors-connectivity/4958/3 solution
    '''
    # take screenshot
    screenshot = plot.plotter.screenshot()

    # The screenshot is just a NumPy array, so we can display it via imshow()
    # and then save it to a file.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot, origin='upper')
    ax.set_axis_off()  # Disable axis labels and ticks
    fig.tight_layout()

    fig.savefig(savepath, dpi=1200)


def main(): 
    # args
    args = input_parse()

    ## PATHS and FILES ## 
    path = pathlib.Path(__file__)

    # raw meg data paths 
    meg_path = path.parents[4] / "834761" / "0108" / "20230928_000000" / "MEG"
    ica_path = path.parents[2] / "data" / "ICA"

    # source reconstruction paths
    bem_path = path.parents[4] / "835482" / "0108" / "bem"
    subjects_dir = path.parents[4] / "835482" 

    # plot path
    plot_path = path.parents[2] / "plots" / "sanity_checks" / "helmet_check"
    plot_path.mkdir(parents=True, exist_ok=True)

    # recordings
    recording_names = {0: '001.self_block1',  1: '002.other_block1',
                       2: '003.self_block2',  3: '004.other_block2',
                       4: '005.self_block3',  5: '006.other_block3'}

    chosen_recording = recording_names[args.recording]

    ## INTIAL PREPROCESSING ## 
    # print ica components to exclude
    ica_components = ica_dict()
    
    # load and preprocess data
    processed_raw = preprocess(meg_path, chosen_recording, ica_path, ica_components[chosen_recording])

    ## EPOCHING ##
    # get events
    events  = mne.find_events(processed_raw, min_duration = 2/processed_raw.info["sfreq"])

    # check if self or other block to determine event ids
    if "self" in chosen_recording: 
        event_id = dict(self_positive=11, self_negative=12, button_img=23)
    else: 
        event_id = dict(other_positive=21, other_negative=22, button_img=23)

    # do epochs 
    reject = dict(mag=4e-12, grad=4000e-13) # T, T/m, V
    epochs = epoching(processed_raw, events, event_id=event_id, tmin=-0.200, tmax=1.000, reject_criterion=reject)

    ## SOURCE RECONSTRUCTION ##
    fwd_name = f'{chosen_recording[4:]}-oct-6-src-5120-fwd.fif'
    fwd = mne.read_forward_solution(bem_path / fwd_name)
    src = fwd['src'] # where are the sources
    trans = fwd['mri_head_t'] # what's the transformation between mri and head
    info = epochs.info # where are the sensors?
    bem_sol = fwd['sol'] # how do electric fields spread from the sources inside the head?

    # plot source space
    bem = bem_path / "0108-5120-bem.fif"

    # plot 
    alignment_plot = mne.viz.plot_alignment(info, trans=trans, subject='0108',
                        subjects_dir=subjects_dir, src=src,
                        bem=bem, dig=True, mri_fiducials=True)
    
    # set view (angle from the side)
    mne.viz.set_3d_view(alignment_plot, 45, 90, distance=0.6, focalpoint=(0., 0., 0.))

    save_3D_figure(alignment_plot, plot_path / f"alignment_{chosen_recording}.png")

if __name__ == "__main__":
    main()