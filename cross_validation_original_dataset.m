%% Cross Validation Script
% Ethan Edmunds, Nov 2024

% Define image folder name
imageFolder = "defect_dataset";

% Collect data from image folder
imds = imageDatastore(imageFolder, "IncludeSubfolders", true, "LabelSource", "foldernames");

% Show the number of each type of class
tbl_count = countEachLabel(imds);

totalImgCount = sum(tbl_count{:,"Count"}); % Print the total number of defects found

% Randomly translate the images up to three pixels horizontally and
% vertically, rotate the images with an angle up to 180 degrees
imageAugmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandYReflection=true, ...
    RandRotation=[-180 180], ...
    RandScale=[0.9 1.1]);

% Define image size
imageSize = [200 200];

noBlocks = 8; % Number of blocks
noFilters = 26; % Number of filters

hyperParams = [noBlocks noFilters]; % Initialize hyperparameters

noClasses = 4; % Number of classes
filterSize = 5; % Filter size

auxParams = [noClasses filterSize]; % Initialize aux parameters

% Create network architecture
networkArchitecture = createNetworkModel(hyperParams, auxParams, imageSize); % Create network architecture to be trained

savePath = 'originalDatasetModelInformation'; % Name of folder to save model information to

% Initiate for loop to randomly select training and validation datasets
for i = 5:10
    
    fprintf('Starting iteration ' + string(i) + '\n')

    % Randomly create training and validation datasets
    [imdsTrain, imdsVal] = splitEachLabel(imds, 0.7, 0.3, "randomized");
    
    % Augment the training and validation datasets
    imdsTrain_aug = augmentedImageDatastore(imageSize, imdsTrain, "DataAugmentation", imageAugmenter);
    imdsVal_aug = augmentedImageDatastore(imageSize, imdsVal, "DataAugmentation", imageAugmenter);
    
    % Training options
    % Note: may need to change execution environment depending on your computer specs 
    networkTraining_options = trainingOptions('adam', ...
        "MiniBatchSize", 64, ...
        'InitialLearnRate', 0.0034, ...
        'MaxEpochs', 10, ...
        'Shuffle','every-epoch', ...
        'ValidationData', imdsVal_aug, ...
        'ValidationFrequency', 5, ...
        'Verbose',true, ...
        'Plots','training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'OutputNetwork', 'best-validation', ...
        'LearnRateSchedule', 'none', ...
        'LearnRateDropFactor', 0.1088);
    
    % Train the network, returning the network model and network info
    [neuralNetwork_model, neuralNetwork_info] = trainNetwork(imdsTrain_aug, networkArchitecture, networkTraining_options); % Train the model

    % Create network predictions for classification
    [neuralNetwork_predictions, ~] = classify(neuralNetwork_model, imdsVal_aug);
    
    % Calculate total accuracy by evaluate
    totalAccuracy = mean(neuralNetwork_predictions == imdsVal.Labels);

    fprintf("\nTotal accuracy for this iteration: " + string(totalAccuracy));
    
    fileName = 'model_' + string(i) + '_info.mat';

    fullFileName = fullfile(savePath, fileName); % Create full path to save file to
    
    save(fullFileName, 'neuralNetwork_model', 'neuralNetwork_info', 'totalAccuracy') % Save the file to the directory
    
    close(findall(groot, 'Tag', 'NNET_CNN_TAININGPLOT_UIFIGURE'));

    fprintf("\n\nModel trained and written to output file | Iteration " + string(i) + " \n\n");

end

% Function to create the neural network architecture
function networkArchitecture = createNetworkModel(hyper_params, aux_params, imageSize)
    
    num_blocks = hyper_params(1); % number of layers to be used
    num_filters = hyper_params(2); % number of filters to be used

    num_classes = aux_params(1); % number of classes to be used
    filter_size = aux_params(2); % size of filters to be used in network design

    networkArchitecture = [imageInputLayer([imageSize 1])]; % Create the input layer for the CNN

    % Create the blocks for the CNN architecture each of which contain
    % convolution -> batch normalisation -> relu -> max pooling layers
    for i = 1:(num_blocks)
        networkArchitecture = [networkArchitecture
            convolution2dLayer(filter_size, num_filters, "Padding", "same")
            batchNormalizationLayer
            reluLayer
            maxPooling2dLayer(2, "Stride", 2, "Padding", "same")
        ];
    end

    % Output layers
    networkArchitecture = [networkArchitecture
        dropoutLayer(0.3)
        fullyConnectedLayer(num_classes)
        softmaxLayer
        classificationLayer
    ];

end