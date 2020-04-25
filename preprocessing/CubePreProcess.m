clear all;clc;
% this code is based upon 'data processing' in FFCC
% using this code to pre-process for cube+ and cube dataset
SetPath = 'G:\cubeDataset';
images_folder = fullfile( SetPath, 'cube_PNG'); % path to the png files of Cube\+ dataset
illuminants_filename = fullfile( SetPath , 'cube_gt.txt');
x_coord = 1050;
y_coord = 2050;
black_level = 2048;

SATURATION_SCALE = 0.95;
gt_rgb = textread(illuminants_filename);
dirents = dir(fullfile(images_folder, '*.png'));
  
% init data
data.exifs = {};
data.images = {};
data.mat_imgs = {};
data.filenames = {};
data.illuminants = {};

% Write each project's images to disk.
output_folder = fullfile( '.\database\Cube' , 'preprocessed_512');
mkdir(output_folder);

for i_file  = 1: length(dirents)
    fprintf('processing image: %5g/%5g \n',i_file,length(dirents));
    % load filename
    image_filename = fullfile(images_folder, dirents(i_file).name);
    root_filename  = dirents(i_file).name(1:end-4);
    data.filenames{i_file} = image_filename;
    % load gt illuminants
    illuminant = gt_rgb( str2num(root_filename),:);
    illuminant = illuminant ./ sqrt( sum(illuminant.^2));
    data.illuminants{i_file} = illuminant;
    % load image
    image  = double(imread(image_filename));
%     close all;imshow(image./65535);
    
    height = size(image,1);
    width  = size(image,2);
    mask   = ones(height,width,'logical'); 
    mask(x_coord:end,y_coord:end) = 0;

   
    image = max(image-black_level,0);
    max_intensity = max(image(:));
    index = find( image(:) >= max_intensity -2);
    image(index) = max_intensity -2;
    image = image./max(image(:)); % normalized to 0-1

    % Update the mask to ignore saturated pixels.
    image(repmat(~mask, [1,1,3])) = 0;
  
    % Modify "image" and "mask" according to each project's specifications.
    % downsample the image file for fast reading in MDLCC training
    stats_size   = 2*[256 384]; 
    bit_width    = 16;
    linear_stats = 1;

    image_project = image;
    mask_project  = mask;
    illuminant_project = illuminant;

    if ~isnan(stats_size)
        % If the image and the stats have inverted aspect ratios, rotate the
        % image and mask.
        if (sign(log(size(image_project,1) / size(image_project,2))) ...
                            ~= sign(log(stats_size(1) / stats_size(2))))
            image_project = cat(3, ...
                rot90(image_project(:,:,1)), ...
                rot90(image_project(:,:,2)), ...
                rot90(image_project(:,:,3)));
            mask_project = rot90(mask_project);
        end

        % Crop off the last row/column of the image so its size is divisible by 2,
        % if necessary.
        image_project = image_project(...
                1:floor(size(image_project,1)/2)*2, ...
                1:floor(size(image_project,2)/2)*2, :);

        mask_project = mask_project(...
                1:floor(size(mask_project,1)/2)*2, ...
                1:floor(size(mask_project,2)/2)*2, :);

        % Crop the image/mask to match the aspect ratio of the specified stats.
        im_size = [size(image_project,1), size(image_project,2)];
        scale   = min(im_size ./ stats_size);
        crop_size = floor( (stats_size * scale)/2 ) * 2;
        image_project = image_project(...
                (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
                (im_size(2) - crop_size(2))/2 + [1:crop_size(2)], :);
        mask_project = mask_project(...
                (im_size(1) - crop_size(1))/2 + [1:crop_size(1)], ...
                (im_size(2) - crop_size(2))/2 + [1:crop_size(2)]);
        im_size = [size(image_project,1), size(image_project,2)];
    end
    
    % using CCM to project image to camera raw color space
    image_canonized = image_project;
    illuminant_canonized = illuminant_project;


    if ~isnan(stats_size)
      % Check that the image and the stats have the same aspect ratio.
      assert(abs(log(im_size(2) / im_size(1)) ...
               - log(crop_size(2) / crop_size(1))) < 0.01)

      % Downsample the image according to the mask, and downsample the mask.
      % This downsample is weighted (or equivalently, done in homogenous
      % coordinates) so that masked pixels are ignored.
      downsample = @(x)imresize(x, stats_size, 'bilinear');

      image_numer = downsample(bsxfun(@times, image_canonized, mask_project));
      image_denom = downsample(double(mask_project));
      image_down  = bsxfun(@rdivide, image_numer, max(eps, image_denom));

      % A small denominator means that very few masked pixels are in the
      % full-res image at that position, and so the low-res image should be
      % masked out at that position.
      DENOM_THRESHOLD = 0.01;
      mask_down = image_denom >= DENOM_THRESHOLD;
    else
      image_down = image_canonized;
      mask_down = mask_project;
    end

    % Zero out the masked values.
    image_down(repmat(~mask_down, [1,1,3])) = 0;

    clear mask_down;
    max_val    = max(image_down(:));
    image_down = min(1, image_down ./ max_val);
    data.mat_imgs{i_file} = single( image_down);
    % Convert the image to the specified bit depth, and cast accordingly.
    image_down = round((2^bit_width-1) * image_down);
    if bit_width <= 8
      image_down = uint8(image_down);
    elseif bit_width <= 16
      image_down = uint16(image_down);
    else
      assert(0);
    end

    data.images{i_file} = image_down;
    data.illuminants{i_file} = illuminant_canonized;

    filename = data.filenames{i_file};
    write_name = str2num(filename( find(filename =='\',1,'last')+1:end-4));
    image_filename = fullfile(output_folder, num2str(write_name, '%06d.png'));
    imwrite(data.images{i_file}, image_filename);
    im = data.mat_imgs{i_file};
    save(fullfile(output_folder, num2str(write_name, '%06d.mat')), 'im');

    illuminants_filename = fullfile(output_folder, num2str(write_name, '%06d_gt.mat'));
    gt = data.illuminants{i_file};
    save(illuminants_filename, 'gt')



    filename_filename = ...
      fullfile(output_folder, num2str(write_name, '%06d_filename.txt'));
    fid = fopen(filename_filename, 'w');
    fprintf(fid, '%s', data.filenames{i_file});
    fclose(fid);
end

