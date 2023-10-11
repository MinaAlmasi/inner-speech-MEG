'''
Functions for classification used in classify.py
'''
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold

from tqdm import tqdm 
import numpy as np
import matplotlib.pyplot as plt

## FUNCTIONS USED IN SIMPLE_CLASSIFICATION FUNCTION
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
        print(pair)
        combine_pair = int(str(pair[0]) + str(pair[1]))
        for i, trigger in enumerate(y_combined):
            if trigger in pair:
                y_combined[i] = combine_pair
    
    return y_combined

## SIMPLE CLASSIFICATION FUNCTION
def simple_classification(X, y, triggers, penalty='none', C=1.0, combine=None):
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

    logr = LogisticRegression(penalty=penalty, C=C, solver='newton-cg')
    
    # scale data 
    sc = StandardScaler() # especially necessary for sensor space as
                          ## magnetometers
                          # and gradiometers are on different scales 
                          ## (T and T/m)

    # init cross validation
    cv = StratifiedKFold()
    
    # get mean scores
    mean_scores = np.zeros(n_samples)
    
    for sample_index in tqdm(range(n_samples)):
        this_X = X[:, :, sample_index]
        sc.fit(this_X)
        this_X_std = sc.transform(this_X)
        scores = cross_val_score(logr, this_X_std, y, cv=cv)
        mean_scores[sample_index] = np.mean(scores)
        
    return mean_scores

def plot_classification(times, mean_scores, title=None, savepath=None):
    fig, ax = plt.subplots()
    ax.plot(times, mean_scores)
    ax.hlines(0.50, times[0], times[-1], linestyle='dashed', color='k')
    ax.set_ylabel('Proportion classified correctly')
    ax.set_xlabel('Time (s)')
    
    # if the title is NOT none, add it! 
    if title:
        ax.set_title(title)

    # if the savepath is NOT none, save the plot
    if savepath: 
        fig.savefig(savepath)

    return fig, ax
