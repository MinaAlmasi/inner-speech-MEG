# Inner Speech MEG
MEG ACN portfolio 3 by study group 7. All code in the `src` folder was utilised for various stages of the project. 

## Structure
The repository is structured as such: 

```
insert tree here
```
Note that the freesurfer and MEG data needs to be placed to folders out in a structure as such (to match the  structure within UCLOUD): 
````

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

