% this code is based upon project FFCC
%
function [image, mask, camera_name, illuminant, CCM] = CCDReadImage(image_filename,...
                               images_folder,coordinates_folder,illuminants_filename)

    % paths = DataPaths;
    SATURATION_SCALE = 0.95;


    image = double(imread(image_filename));

    % loading camera index
    if strcmp( image_filename(find(image_filename == '\', 1, 'last') + [1:3]), 'IMG')
        camera_name = 'Canon5D';
    else
        camera_name = 'Canon1D';
    end

    % loading sensor setting
    gehlershi_sensor = LudyGehlerShiSensor;
    black_level      = gehlershi_sensor.BLACK_LEVEL.(camera_name);
    saturation       = gehlershi_sensor.SATURATION.(camera_name);
    root_filename    = image_filename((find(image_filename == '\', 1, 'last') + 1) : end-4);

    % loading groundtruth illuminants
    illuminants = load(illuminants_filename);
    dirents = dir(fullfile( images_folder ,'*.png'));
    dirents = {dirents.name};
    dirents = dirents(cellfun(@(x)(x(1) ~= '.'), dirents));
    assert(length(dirents) == size(illuminants.real_rgb,1));
    filename = image_filename((find(image_filename == '\', 1, 'last') + 1) : end);
    i_file = find(cellfun(@(x) strcmp(x, filename), dirents));
    assert(length(i_file) == 1)
    illuminant = double(illuminants.real_rgb(i_file,:));
    illuminant = illuminant(:) ./ sqrt(sum(illuminant.^2));

    % loading checker board mask
    cc_coord = load(fullfile(coordinates_folder, [root_filename '_macbeth.txt']));
    scale    = cc_coord(1, [2 1]) ./ [size(image,1) size(image,2)];
    mask     = ~roipoly(image, cc_coord([2 4 5 3],1) / scale(1), cc_coord([2 4 5 3],2) / scale(2));


% Subtract out the black level and adjust the saturation value accordingly.
image = max(image-black_level,0);
saturation = saturation - black_level;

% Scale and clamp the image.
image = min(1, image / (SATURATION_SCALE * saturation));

% Update the mask to ignore saturated pixels.
% mask = mask & all(image < 1, 3);
