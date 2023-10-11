'''
Script to classify brain areas. 

Run in the terminal: 
    python src/classify.py -area {BRAIN_LABEL_TO_CLASSIFY}
'''

import pathlib
import argparse

# custom modules with functions!  
from utils.classify_preprocess import preprocess_sensor_space_data, preprocess_source_space_data
from utils.classify_fns import simple_classification, plot_classification


def input_parse(): 
    parser=argparse.ArgumentParser()

    # add arguments to parser
    parser.add_argument("-label", "--brain_label", type=str, help="brain label to classify on (from freesurfer)", default="rh.supramarginal.label")
    args = parser.parse_args()

    return args

def main(): 
    args = input_parse()
    label_to_classify = args.brain_label

    # define paths
    path = pathlib.Path(__file__)
    raw_path = path.parents[3] / "834761" # path.parents[i] denotes how many steps you go out from where the .py file is located (i.e., number of ".." / ".." in a file path)
    subjects_dir = path.parents[3] / "835482" 
    
    plots_path = path.parents[1] / "plots"
    plots_path.mkdir(parents=True, exist_ok=True) # make plots path if it does not exist

    # preprocess SENSOR space
    epochs_list = preprocess_sensor_space_data(
        '0114', '20230927_000000',
        raw_path=raw_path,
        decim=10,
        reject_criterion=dict(mag=4e-12, grad=4000e-13, eog=250e-6)
        ) # don't go above decim=10

    # preprocess SOURCE space 
    brain_area, y = preprocess_source_space_data(
        subject = '0114', 
        date = '20230927_000000',
        
        raw_path=raw_path,
        subjects_dir=subjects_dir,

        label=label_to_classify,
        epochs_list=epochs_list
        )

    # perform classification
    triggers = [11, 12]

    classification = simple_classification(
                                X=brain_area, 
                                y=y, 
                                triggers=triggers,
                                penalty='l2', 
                                C=1e-3, 
                                combine=None
                                )

    # plot classification
    times = epochs_list[0].times  # extract time for plotting

    plot_classification(
        times = times, 
        mean_scores = classification, 
        title = f"{label_to_classify}. Triggers: {triggers}",
        savepath = plots_path / f"{label_to_classify}.png"
    )

if __name__ == "__main__": 
    main()

