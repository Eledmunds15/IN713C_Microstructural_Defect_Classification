%% Image Processing: Defect Detection and Dataset Development
% Ethan L. Edmunds, Nov 2024

image_num = 18; % total number of micrographs to analyse
output_folder = 'unsorted_images_test'; % identify the output folder

% check if the output_folder has been created or not
if ~exist(output_folder, 'dir')
    mkdir(output_folder)
end

% Iterate through the array of indexes and apply the processImage function
for i = 1:image_num
    processImage(i, output_folder)
end

% A function that takes the index of an image and processes it
function m = processImage(image_no, output_folder)
    
    % Identify the image that needs to be processed
    image_name = "image_"; % identify the image name
    image_format = ".tif"; % identify the format of the image
    directory = "IN713C_original_micrographs\\"; % identify the parent directory which stores all of the images
    full_name = directory + image_name + image_no + image_format; % Create full name for image file to be downloaded
    
    % Download and show the image
    img = imread(full_name);

    img_gs = rgb2gray(img); % Grayscale the image
    img_gs = img_gs < 80; % Using thresholding to identify pixels with a brightness less than that of the threshold

    [rows, cols] = size(img_gs); % Find the size of the image

    img_gs(round(rows*0.95):end , : ) = []; % Crop out the bottom 5% of the image to remove the scale

    img_gs = imclearborder(img_gs); % Get rid of all images on the border
    img_gs = bwareafilt(img_gs, [15 inf]); % Filter out small defects with an area of less than 20

    def_ds = regionprops(img_gs, "Image", "BoundingBox", "Area"); % Creates a datastore containing information and an image of each defect

    % Sort the datastore based on area
    def_ds_table = struct2table(def_ds)
    def_ds_table = sortrows(def_ds_table, "Area", "ascend");
    def_ds = table2struct(def_ds_table);

    fprintf("The number of defects in the image after filtering is: " + length(def_ds) + "\n\n");

    % Iterate through each defect in the image and download them
    % figure
    for i = 1:length(def_ds)
        
        % Show each individual defect in the micrograph
        target_def = def_ds(i).Image;
        % subplot(1,1,1), imshow(target_def, "Border", "loose")

        % Create the defect image name
        defectImage_name = output_folder + "\\" + image_name + image_no + "_" + string(i) + ".tif";

        % Write the image to a file
        imwrite(target_def, defectImage_name);
        
        fprintf("\nImage " + image_name + image_no + "_" + string(i) + ".tif successfully downloaded...\n")

    end
    
    m = 0;
    fprintf("Image " + image_no + " has been processed...\n\n")

end