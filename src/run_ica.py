import pathlib
import mne

def main():
    # define paths 
    path = pathlib.Path(__file__)
    meg_path = path.parents[3] / "834761" / "0108" / "20230928_000000" / "MEG"

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

        # crop to remove initial HPI noise and noise at the end of each trial (verified by manually checking raws in run_raw.py)
        cropped = raw.copy().crop(tmin=10, tmax=365)
        del raw

        # some initial filtering 
        filtered = cropped.copy().filter(l_freq=1, h_freq=40)
        filtered.apply_proj()

        resampled = filtered.copy().resample(250)
        del filtered

        # do ICA 
        ica = mne.preprocessing.ICA(n_components=0.9999, random_state=42, max_iter=3000)

        # fit ICA
        ica.fit(resampled)

        # save ICA 
        ica_outpath = path.parents[1] / "data" / "ICA"
        ica_outpath.mkdir(parents=True, exist_ok=True)
        ica.save(ica_outpath / f"{name}-ica.fif", overwrite=True)

        # plot & save components
        components = ica.plot_components(show=False)

        # saving components
        comp_path = path.parents[1] / "plots" / "ICA" / name
        comp_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

        # unzip components and save each component separately
        for i, component in enumerate(components):
            component.savefig(comp_path / f"component_{i}.png")

        # get sources 
        source_path = path.parents[1] / "plots" / "ICA" / "sources" / name
        source_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

        # saving source
        batch_size = 20

        with mne.viz.use_browser_backend('matplotlib'):
            for start_pick in range(0, ica.n_components_, batch_size):
                end_pick = min(start_pick + batch_size, ica.n_components_)
                sources = ica.plot_sources(resampled, show=False, show_scrollbars=False, picks=(range(start_pick, end_pick)))
                sources.savefig(source_path / f"sources_{start_pick}_{end_pick}.png")

if __name__ == "__main__":
    main()