# Inner Speech MEG
Investigating inner speech using MEG. Portfolio 3 by study group 7 (Alina, Anton, Juli, Martine, Mina) in the course *Advanced Cognitive Neuroscience* (E2023). 

All code in the `src` folder was utilised for various stages of the project. 

Below, an overview of the repository is described. For reproducing the results, please follow the *Pipeline* section. Please note that the data cannot be made publicaly available, and the scripts are therefore not able to run without gaining access to the data.

## Structure
The repository is structured as such: 

```
├── LICENSE
├── README.md
├── data
│   ├── ICA                   <--- place trained ICA fits to apply to data at a later point
│   └── README.md
├── env_to_ipynb_kernel.sh    
├── figures                   <---- figures used in paper
├── nbs                       <---- notebooks for testing, can be ignored
├── plots                     <---- plots used for figure creation
├── requirements.txt
├── setup.sh                  <---- run to install reqs in env
└── src 
    ├── classify.py           <---- for classifiers on source space
    ├── run_ica.py            <---- fit and plot ICA components
    ├── run_raw.py            <---- visualise raw data w. intial preprocesing (to crop data sensibly)
    ├── sanity_checks         <---- several scripts for sanity checking
    ├── stc_plot.py           <---- for plotting source time courses
    └── utils                 <---- contains helper functions for main scripts

```
Note that the freesurfer and MEG data needs to be placed in a structure as such (to match the structure within UCLOUD) for the paths to work when running the code: 
````
├── 834761     <---- MEG data
│   └── 0XXX   
├── 835482     <---- freesurfer data
│   ├── 0XXX   <---- participant number
│   └── fsaverage
├── code
│   └── inner-speech-MEG   <---- code repository
````

## Event Triggers
For the analysis, the following event triggers are relevant to know: 
|       Desc.        |   Trigger   |
|------------------|-----------|
|     IMG_PS       |    11     |
|     IMG_PO       |    21     |
|     IMG_NS       |    12     |
|     IMG_NO       |    22     |
|     IMG_BI       |    23     |
|  button_press    |   202     |


Importantly, the letter at the end of trigger denotes the condition `self (S)` and `other (O)`, and whether the stimuli is `positive (P)` or `negative (N)`. 
For instance, `IMG_PS` refers to the the **positive** stimuli in the **self** condition. 

The trigger `IMG_BI` refers to the stimuli connected with a `button_press`. 

## Pipeline
Firstly, please install all necessary requirements by typing in the terminal: 
```
bash setup.sh 
```
To run any code, please remember to firstly activate your virtual environment by typing `source env/bin/activate` in your terminal while being in the main folder of the directory (`cd inner-speech`).

### Running the code
#### Sanity Checks
To run a sanity check, type in the terminal (while being in the main folder):
```
python src/sanity_checks/<FILE_I_WANT_TO_RUN>
```
For instance, you can run the check of visual activation by typing
```
python src/sanity_checks/helmet_check.py
```

#### Other analysis
To run the classification or any other file within the `src` folder, type (while being in the main folder):
```
python src/<FILE_I_WANT_TO_RUN>
```

For instance, you can train and plot ICA components by typing: 
```
python src/run_ica.py
```