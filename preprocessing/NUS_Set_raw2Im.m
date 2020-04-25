clear all
clc
% this code is based upon 'data processing' in FFCC

nus_path   = './database/NUS';
write_path = './database/NUS';
load(fullfile(nus_path,'projects'));

SATURATION_SCALE = 0.95;
bit_width = 16;



for i = 1:  length( project_names )
    
    % init data
    data.images = {};
    data.filenames = {};
    data.illuminants = {};
    data.im = {};
    folder_name = [ 'Cheng' project_names{i}];
    project   = project_names{i};
    
    im_folder = fullfile('F:\ChengDataset', project,'PNG');
    gt_name   = fullfile(nus_path,folder_name,[project '_gt.mat']);
    ccm_name  = fullfile(nus_path,'tags',[project '_CCM.txt']);
    im_list      = dir( fullfile(im_folder,'*.png'));
    info         = load( gt_name );
    
    
    for i_file = 1:length( im_list )
        disp( i_file);
        illuminant = info.groundtruth_illuminants(i_file,:);
        illuminant = illuminant(:) ./ sqrt(sum(illuminant.^2));
        im_name = fullfile( im_folder, im_list(i_file).name );
        im   = double(imread( im_name ));

        black_level    = info.darkness_level;
        saturation     = info.saturation_level;
        root_filename  = im_name((find(im_name == '\', 1, 'last') + 1) : end-4);
    
        mask = true(size(im,1), size(im,2));
        mask(info.CC_coords(i_file,1) : info.CC_coords(i_file,2), info.CC_coords(i_file,3) : info.CC_coords(i_file,4), :) = false;
        im(repmat(~mask, [1,1,3])) = 0;
        
        % Subtract out the black level and adjust the saturation value accordingly.
        im = max(im-black_level,0);
        saturation = saturation - black_level;
        
        % Scale and clamp the image.
        im = min(1, im / (SATURATION_SCALE * saturation));
%         mask = mask & all(im < 1, 3);
        
        %-----------------------------------------------
        % Modify "image" and "mask" according to each project's specifications.
        stats_size   = 2*[256 384];
        bit_width    = 16;
        linear_stats = 1;

        image_project = im;
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

      % Convert the image to the specified bit depth, and cast accordingly.
      im_norm1 = image_down;
      image_down = round((2^bit_width-1) * image_down);
      if bit_width <= 8
          image_down = uint8(image_down);
      elseif bit_width <= 16
          image_down = uint16(image_down);
      else
         assert(0);
      end

      data.images{end+1} = image_down;
      data.im{end+1} = im_norm1;
      data.illuminants{end+1} = illuminant_canonized;
      data.filenames{end+1} = root_filename;
      %-----------------------------------------------
      
    % Write each project's images to disk.
    output_folder = fullfile( nus_path, ['Cheng' project],'preprocessed_512');
    if ~exist(output_folder,'dir')
        mkdir(output_folder);
    end

    image_filename = fullfile(output_folder, num2str(i_file, '%06d.mat'));
    im = single(data.im{i_file});
    save( image_filename, 'im');
    imwrite(data.images{i_file}, fullfile(output_folder, num2str(i_file, '%06d.png')));

    illuminants_filename = fullfile(output_folder, num2str(i_file, '%06d_gt.mat'));
    gt = data.illuminants{i_file};
    save(illuminants_filename, 'gt');



  
    end
end