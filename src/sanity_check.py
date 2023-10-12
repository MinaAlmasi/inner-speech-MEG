# utils 
import pathlib 

# custom module
from utils.general_preprocess import combine_raws, create_evoked

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
    
    # compute raws for all recordings 
    reject = dict(mag=4e-12, grad=4000e-13, eog=250e-6) # T, T/m, V
    epochs = combine_raws(meg_path, recording_names, reject_criterion=reject)

    print(epochs)

    plot = epochs.plot_image(picks="meg")

    # save plot
    plot.savefig(plots_path / "epochs.png")

    # compute evoked 
    evoked = create_evoked(epochs, triggers=['positive', 'negative', 'button_img'])

    print(evoked)


if __name__ == "__main__":
    main()