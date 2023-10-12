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

    # load raw
    raw = mne.io.read_raw(meg_path / recording_names[0] / 'files' / "self_block1.fif", preload=True)

    # pick types
    raw.pick_types(meg=True, eeg=False, stim=True)

    # filter raws
    raw.filter(h_freq=40, l_freq = 0.1, n_jobs=4) # alters raw in-place

    # apply projs
    raw.apply_proj()

    # plot raw
    mne.viz.plot_raw(raw, duration=20, n_channels=30, block=True)


if __name__ == "__main__":
    main()