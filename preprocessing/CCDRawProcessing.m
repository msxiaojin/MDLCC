clear all;clc
% this code is based upon 'data processing' in FFCC


ccdSetPath = './database/CCD';
% download png images, mask info and GT
% from website: https://www2.cs.sfu.ca/~colour/data/shi_gehler/
images_folder = fullfile( ccdSetPath, 'png');
coordinates_folder = fullfile( ccdSetPath, 'ColorCheckerDatabase_MaskCoordinates/coordinates');
illuminants_filename = fullfile( ccdSetPath , 'real_illum_568..mat');
crossvalidationfolds_filename = fullfile(ccdSetPath, './threefoldCVsplit.mat');

SATURATION_SCALE = 0.95;
load(illuminants_filename);
dirents = dir(fullfile(images_folder, '*.png'));

load(crossvalidationfolds_filename);

% Assert that the filenames are in the same order as is expected by the
% cross-validation struct.
roots = {Xfiles.name};
roots = cellfun(@(x) x(1:find(x=='.')-1), roots, 'UniformOutput', false);


% Turn the cross-validation test set splits into a single fold index per image.
is_test = [sparse(te_split{1}, 1, true, 568, 1), ...
       sparse(te_split{2}, 1, true,568, 1), ...
       sparse(te_split{3}, 1, true, 568, 1)];
assert(all(sum(is_test,2) == 1))
fold_idx = uint8(is_test * [1:3]');

% Write each project's images to disk.
output_folder = fullfile( ccdSetPath , 'full_preprocessed');
mkdir(output_folder);


for i_file  = 1:  length(dirents)
    
    im_filename = fullfile(images_folder, dirents(i_file).name);
    data.filenames{i_file} = im_filename;

    [image, mask, camera_name, illuminant] = CCDReadImage(im_filename,...
                               images_folder,coordinates_folder,illuminants_filename);

    image(repmat(~mask, [1,1,3])) = 0;
    root_filename = dirents(i_file).name;
    root_filename = root_filename(1:(find(root_filename == '.', 1, 'first')-1));
  
    image(repmat(~mask, [1,1,3])) = 0;

    max_val    = max(image(:));
    disp(max_val)
    image = min(1, image ./ max_val);
    im_norm1 = image;

    % Convert the image to the specified bit depth, and cast accordingly.
    bit_width = 16;
    image = round((2^bit_width-1) * image);
    image = uint16(image);
    

    %% writting to disk
    image_filename = fullfile(output_folder, num2str(i_file, '%06d.mat'));
    im = single(im_norm1);
    save(image_filename,'im');
    imwrite(image, fullfile(output_folder, num2str(i_file, '%06d.png')));

    illuminants_name = fullfile(output_folder, num2str(i_file, '%06d_gt.mat'));
    gt = illuminant;
    save(illuminants_name, 'gt');
end

