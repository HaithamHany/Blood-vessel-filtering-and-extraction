% Define the folders for the images and ground truths
imageFolder = 'images/';
gtFolder = 'groundTruth/';

% Get the list of images and ground truths
imageFiles = dir(fullfile(imageFolder, '*.tif')); 
gtFiles = dir(fullfile(gtFolder, '*.png'));

% Initialize matrices to store sensitivity (TPR) and false positive rate (FPR)
thresholds = linspace(0, 1, 100); % 100 thresholds between 0 and 1
sensitivityMatrix = zeros(length(imageFiles), length(thresholds)); 
falsePositiveRateMatrix = zeros(length(imageFiles), length(thresholds));

% Loop over each image
for i = 1:length(imageFiles)
    % Read the current image and its ground truth
    I_Original = imread(fullfile(imageFolder, imageFiles(i).name));
    GT = imread(fullfile(gtFolder, gtFiles(i).name));
    GT = GT(:,:,1) > 128; % Binarize and ensure it's logical

    % Process the image to get the continuous output
    vessel_enhanced_MF_FDOG = processImage(I_Original);

    % Calculate metrics for each threshold
    for t = 1:length(thresholds)
        threshold = thresholds(t);
        binaryImage = vessel_enhanced_MF_FDOG >= threshold;
        [sensitivity, specificity] = ground_truth(binaryImage, GT);
        
        sensitivityMatrix(i, t) = sensitivity;
        falsePositiveRateMatrix(i, t) = 1 - specificity;
    end
end

% Average TPR and FPR over all images for each threshold
meanSensitivity = mean(sensitivityMatrix, 1);
meanFalsePositiveRate = mean(falsePositiveRateMatrix, 1);

% Plot the ROC curve
figure;
plot(meanFalsePositiveRate, meanSensitivity, '-o');
xlabel('False Positive Rate (1 - Specificity)');
ylabel('True Positive Rate (Sensitivity)');
title('ROC Curves of MF-FDOG for DRIVE Dataset');
xlim([0, 1]); % Full range of FPR
ylim([0, 1]); % Full range of TPR

% Process Image Function
function vessel_enhanced = processImage(I_Original)
    % Assuming I_Original is already read and is grayscale
    I = im2double(rgb2gray(I_Original));

    % Contrast Enhancement using Adaptive Histogram Equalization
    I_enhanced = adapthisteq(I,'ClipLimit',0.007); 

    % Apply MF filter
    filter_MF = gaussian_matched_filter(0.5, 0.5);
    response_MF = conv2(I_enhanced, filter_MF, 'same');

    % Background subtraction for MF
    background_MF = imopen(response_MF, strel('disk', 15));
    vessel_enhanced_MF = response_MF - background_MF;

    % Apply FDOG filter
    fdog = fdog_filter(0.5);
    response_FDOG = conv2(I_enhanced, fdog, 'same');

    % Background subtraction for FDOG
    background_FDOG = imopen(response_FDOG, strel('disk', 15));
    vessel_enhanced_FDOG = response_FDOG - background_FDOG;

    % Combine responses by addition
    combined_response = vessel_enhanced_MF + vessel_enhanced_FDOG;

    % Normalize the combined response to [0, 1]
    vessel_enhanced = mat2gray(combined_response); % This ensures the output is continuous

end

% Gaussian Matched Filter Function
function f = gaussian_matched_filter(s, L)
    % Define the range for x and y
    x = linspace(-3*s, 3*s, ceil(6*s)+1);
    y = linspace(-L/2, L/2, L+1);

    % Calculate the 2D Gaussian matched filter
    [X, Y] = meshgrid(x, y);
    f = (1/(sqrt(2*pi)*s)) * exp(-X.^2/(2*s^2));
    f = f - mean(f(:)); % Subtract the mean to have zero mean
end

% FDOG Filter Function
function f = fdog_filter(s)
    % Define the range for x
    x = linspace(-3*s, 3*s, ceil(6*s)+1);

    % Calculate the first derivative of Gaussian
    f = -(x/(sqrt(2*pi)*s^3)) .* exp(-x.^2/(2*s^2));
    f = f - mean(f(:)); % Subtract the mean to have zero mean
end

% Ground Truth Function
function [sensitivity, specificity] = ground_truth(binaryImage, GT_binary)
    % Calculate the true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN)
    TP = sum(binaryImage(:) & GT_binary(:));
    TN = sum(~binaryImage(:) & ~GT_binary(:));
    FP = sum(binaryImage(:) & ~GT_binary(:));
    FN = sum(~binaryImage(:) & GT_binary(:));
    
    % Calculate sensitivity and specificity
    sensitivity = TP / (TP + FN);
    specificity = TN / (TN + FP);
end
