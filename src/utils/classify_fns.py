'''
Functions for running classifications on source space data
'''
# utils 
import pathlib
from tqdm import tqdm 
import numpy as np

# MEG package for preprocessing
import mne

# classification models + scaling + evaluation (cross validation)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, permutation_test_score
from sklearn.inspection import permutation_importance

# plotting
import matplotlib.pyplot as plt

## PREPROCESSING 
def get_source_space_data(epochs_dict:dict, subjects_dir, subject:str="0108", label=None, method="dSPM"):
    '''
    Extract source space data for classification 
    (loosely based on https://mne.tools/stable/auto_examples/decoding/decoding_spatio_temporal_source.html#ex-dec-st-source)

    Args
        epochs_dict (dict): dictionary with epochs for each recording (keys are recording names, values are epochs objects)
        subjects_dir (str): path to subjects_dir
        subject (str): subject name (defaults to "0108")
        label (str): label name
    '''
    # set empty array for y
    y = np.zeros(0)

    # extract y for all epochs and concatenate
    for epochs in epochs_dict.values():
        y = np.concatenate((y, epochs.events[:, 2]))

    # load labels if relevant (if None, it will do a whole brain analysis)
    if label is not None:
        label_path = subjects_dir / subject / 'label' / label
        label = mne.read_label(label_path)
    
    for epochs_index, (recording_name, epochs) in enumerate(epochs_dict.items()):
        fwd_name = f"{recording_name[4:]}-oct-6-src-5120-fwd.fif"

        # read forward solution
        fwd = mne.read_forward_solution(subjects_dir / subject / 'bem' / fwd_name)

        # source estimation! 
        noise_cov = mne.compute_covariance(epochs, tmax=0.000)
        
        inv = mne.minimum_norm.make_inverse_operator(epochs.info,
                                                     fwd, noise_cov)
  
        stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inv, lambda2=1,
                                                     method=method, label=label,
                                                     pick_ori="normal")
        # extract source space
        this_X = np.array([stc.data for stc in stcs])

        # concatenate (if first iteration, create X)
        if epochs_index == 0:
            X = this_X
        else:
            X = np.concatenate((X, this_X))
    
    return X, y 

## CLASSIFICATION FUNCTIONS USED IN SIMPLE CLASSIFICATION FUNCTION
def get_indices(y, triggers):
    '''
    Extract indices for triggers
    '''
    indices = list()
    for trigger_index, trigger in enumerate(y):
        if trigger in triggers:
            indices.append(trigger_index)
            
    return indices

def balance_class_weights_multiple(X, y):
    '''
    Balances the class weight by removing trials so each class has the same number of trials as the class with the least trials.

    Args
        X (array): data array with shape (n_channels, n_trials, n_times)
        y (array): contains several classes with shape (n_trials, )

    Returns
        X_equal (array): data array with shape (n_channels, n_trials, n_times) with equal number of trials for each class
        y_equal (array): contains the classes of the original y array, but now with an equal number of trials for each class
    '''
    keys, counts = np.unique(y, return_counts = True)

    keep_inds = []

    for key in keys:
        index = np.where(np.array(y) == key)
        random_choices = np.random.choice(index[0], size = counts.min(), replace=False)
        keep_inds.extend(random_choices)
    
    X_equal = X[keep_inds, :, :]
    y_equal = y[keep_inds]

    return X_equal, y_equal

def combine_triggers(y, combine):
    '''
    Combine triggers for analysis across conditions
    '''
    y_combined = y.copy()

    for pair in combine:
        combine_pair = int(str(pair[0]) + str(pair[1]))
        for i, trigger in enumerate(y_combined):
            if trigger in pair:
                y_combined[i] = combine_pair
    
    return y_combined

## SIMPLE CLASSIFICATION FUNCTION
def simple_classification(X, y, triggers, penalty='none', C=1.0, n_splits=5, combine=None, n_permutations=100):
    '''
    Perform a Logistic regression 
    '''

    n_samples = X.shape[2]

    # get indices for only the triggers we want
    indices = get_indices(y, triggers)

    # reduce data based on these indices
    X = X[indices,:,:]
    y = y[indices]
    
    # equalize data (balance classes, so no triggers are overrepresented )
    X, y = balance_class_weights_multiple(X, y)

    if combine:
        y = combine_triggers(y, combine)

    #clf = LogisticRegression(penalty=penalty, C=C, solver='newton-cg')
    clf = GaussianNB()

    # scale data 
    sc = StandardScaler() # especially necessary for sensor space as
                          ## magnetometers
                          # and gradiometers are on different scales 
                          ## (T and T/m)

    # init cross validation
    cv = StratifiedKFold(n_splits = n_splits, random_state=42, shuffle=True)
    
    # init vals 
    mean_scores = np.zeros(n_samples)
    
    permutation_scores = np.zeros((n_samples, n_permutations))
    y_pred_all = []
    y_true_all = [] 
    
    for sample_index in tqdm(range(n_samples)):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)

        # cross val
        y_pred = cross_val_predict(clf, this_X_std, y, cv=cv)

        scores = np.mean(y_pred == y)
        mean_scores[sample_index] = scores

        y_pred_all.append(y_pred)
        y_true_all.append(y)

        # permutation tst
        _, permutation_score, pvalue = permutation_test_score(clf, this_X_std, y, cv=cv)
        permutation_scores[sample_index, :] = permutation_score
        
    return mean_scores, y_pred_all, y_true_all, permutation_scores

def get_permutation_quantiles(permutation_scores):
    percentile_01 = np.quantile(permutation_scores, 0.01, axis=1)
    percentile_99 = np.quantile(permutation_scores, 0.99, axis=1)

    return percentile_01, percentile_99

def plot_classification(times, mean_scores, permutation_scores, title=None, savepath=None):
    # get permutation quantiles
    percentile_01, percentile_99 = get_permutation_quantiles(permutation_scores)

    # Set figure size for better aspect ratio
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot data in greyscale
    ax.plot(times, mean_scores, 'k-', linewidth=1.5, label='Mean Scores')

    # Plot permutation scores
    ax.fill_between(times, percentile_01, percentile_99, color = "lightgray", alpha=0.55)
    
    # Add a dashed line at y=0.5
    ax.hlines(0.50, times[0], times[-1], linestyle='dashed', color='red', linewidth=0.75)
    
    # Set labels, title and grid
    ax.set_ylabel('Proportion classified correctly', fontsize=14)
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')

    if savepath: 
        fig.savefig(savepath, dpi=1200, bbox_inches='tight')

    return fig, ax
