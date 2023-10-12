import pathlib
import mne

def main():
    # define paths 
    path = pathlib.Path(__file__)
    meg_path = path.parents[3] / "834761" / "0114" / "20230927_000000" / "MEG"

    # define recording names 
    recording_names = ['001.self_block1',  '002.other_block1',
                       '003.self_block2',  '004.other_block2',
                       '005.self_block3',  '006.other_block3']
    # load raw
    for _, name in enumerate(recording_names):
        fif_fname = name[4:]
        full_path = meg_path / name / 'files' / (fif_fname + '.fif')
        raw = mne.io.read_raw(full_path, preload=True)
        raw.load_data()
        raw.pick_types(meg=True, eeg=False, stim=True)

        # remove bad channel  
        raw.info['bads'] += ['MEG0422']
        raw.drop_channels(raw.info['bads'])

        # some initial filtering 
        raw.filter(l_freq=0.1, h_freq=40, n_jobs=4)
        raw.apply_proj()

        # do ICA 
        #ica = mne.preprocessing.ICA(n_components=0.9999, random_state=42, max_iter=3000)
        ica = mne.preprocessing.ICA(n_components=40, random_state=42, max_iter=800)

        # fit ICA
        ica.fit(raw)

        # plot & save components
        components = ica.plot_components(show=False)

        # saving components
        comp_path = path.parents[1] / "plots" / "ICA" / name
        comp_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

        # unzip components and save each component separately
        for i, component in enumerate(components):
            component.savefig(comp_path / f"component_{i}.png")

        # apply ICA 
        raw_copy = raw.copy()
        ica.apply(raw_copy)

        # get sources 
        source_path = path.parents[1] / "plots" / "ICA" / "sources" / name
        source_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

        # saving source
        batch_size = 20

        with mne.viz.use_browser_backend('matplotlib'):
            for start_pick in range(0, ica.n_components_, batch_size):
                end_pick = min(start_pick + batch_size, ica.n_components_)
                sources = ica.plot_sources(raw_copy, show=False, show_scrollbars=False, picks=(range(start_pick, end_pick)))
                sources.savefig(source_path / f"sources_{start_pick}_{end_pick}.png")

if __name__ == "__main__":
    main()