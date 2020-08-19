------------------------Data precessing-----------------------

Gehler-Shi dataset:
1. Download PNG images, ground truth illuminants, cross validation splits and mask of colorcheck 
from https://www2.cs.sfu.ca/~colour/data/shi_gehler/
The GT illuminants and cross validation splits are also provided in this repository.

2. Run the ./preprocessing/CCDRawProcessing.m to pre-process the Gehler-Shi datasets for training and validation.
The preprocessing contains mainly black-level subtraction, saturate-pixel removal, mask out the colorcheck.

---------------------------------

NUS dataset:
1. Download PNG images, ground truth illuminants, and mask of colorcheck 
from http://cvil.eecs.yorku.ca/projects/public_html/illuminant/illuminant.html

2. The subsets for each camera in NUS dataset contain images from the same scene. To ensure that
the same scene would not be in both training and testing sets when combining multiple subsets in the NUS dataset,
we split the training and testing set for NUS dataset according to scene content. 

The training and testing splits of the NUS dataset are provides in ./database/NUS/cvsplits_nus.mat

3. Run the ./preprocessing/NUS_Set_raw2Im.m to pre-process the NUS dataset

----------------------------------

Cube+ dataset:
1. 

