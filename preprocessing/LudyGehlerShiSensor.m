% this code is based upon project FFCC

function gehlershi_sensor = LudyGehlerShiSensor

% The blacks levels for the two cameras used by the Gehler-Shi dataset.
gehlershi_sensor.BLACK_LEVEL.Canon1D = 0;
gehlershi_sensor.BLACK_LEVEL.Canon5D = 129;
% The saturation levels used by the two cameras in the dataset, which happen to
% be the same. Past papers use 3580 as the saturation value, which does not
% appear to be correct.
gehlershi_sensor.SATURATION.Canon1D = 3692;
gehlershi_sensor.SATURATION.Canon5D = 3692;
% Our estimate of the CCMs of the two cameras used in the dataset. This is not
% used for training models for this project, but is useful for preprocessing.
% working_dir = pwd;

