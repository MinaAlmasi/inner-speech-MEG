import pathlib
import mne
import matplotlib.pyplot as plt

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

    # plot raw 
    raw.filter(l_freq=None, h_freq=30, n_jobs=4) # alters raw in-place
    mne.viz.plot_raw(raw, duration=10, n_channels=30, scalings="auto", block=True)

    # save plot
    #plot[0].savefig(plots_path / "raw.png")


if __name__ == "__main__":
    main()