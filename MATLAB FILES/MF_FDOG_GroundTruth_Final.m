I_Original = imread('21_training.tif');
if size(I_Original, 3) == 3
    I = rgb2gray(I_Original); % Convert to grayscale if the image is RGB
end

% Contrast Enhancement using Adaptive Histogram Equalization
I_uint8 = im2uint8(I); % Convert to uint8 for adapthisteq
I_enhanced_uint8 = adapthisteq(I_uint8,'ClipLimit',0.007); 
I = im2double(I_enhanced_uint8); % Convert back to double for further processing


% Parameters for filters
s = 0.5; % Scale for Gaussian
L = 0.5; % Length for smoothing

% Apply MF filter
filter_MF = gaussian_matched_filter(s, L);
response_MF = conv2(I, filter_MF, 'same');

% Background subtraction for MF
se = strel('disk', 15);
background = imopen(response_MF, se);
vessel_enhanced_MF = response_MF - background;

%Apply FDOG filter
fdog = fdog_filter(s);
response_FDOG = conv2(I, fdog, 'same');


% Background subtraction for FDOG
se = strel('disk', 15);
background_FDOG = imopen(response_FDOG, se);
vessel_enhanced_FDOG = response_FDOG - background_FDOG;

% Calculate the local mean of the FDOG response
w = 15; % Size of local mean filter
W = ones(w) / w^2;
Dm = conv2(vessel_enhanced_FDOG, W, 'same');
m_D = (Dm - min(Dm(:))) / (max(Dm(:)) - min(Dm(:)));

% Set reference threshold Tc
c = 1.5; % The constant c between 2 and 3
mu_h = mean(vessel_enhanced_MF(:));
Tc = c * mu_h;

% Adjust the threshold T based on m_D
T = (1 + m_D) * Tc;

% Thresholding the response to MF
T_MF = Tc; % Simple threshold based on mean value of H
binary_MF = vessel_enhanced_MF >= T_MF;

% Apply combined thresholding scheme using the responses from MF-FDOG
binary_MF_FDOG = vessel_enhanced_MF >= T;

% Post-processing to remove noise and close gaps in the vessels
P = 50; % Minimum area threshold to remove small objects
se = strel('disk', 1); % Structuring element for closing

% Remove small objects from binary images
binary_MF_clean = bwareaopen(binary_MF, P);
binary_MF_FDOG_clean = bwareaopen(binary_MF_FDOG, P);

% Close small gaps within the vessels
binary_MF_closed = imclose(binary_MF_clean, se);
binary_MF_FDOG_closed = imclose(binary_MF_FDOG_clean, se);

% Display results
figure;
subplot(3,2,1), imshow(I_Original, []), title('Original Image');
subplot(3,2,2), imshow(vessel_enhanced_MF, []), title('Response by MF');
subplot(3,2,3), imshow(vessel_enhanced_FDOG, []), title('Response by FDOG');
subplot(3,2,4), imshow(binary_MF_closed, []), title('Thresholded MF Response');
subplot(3,2,5), imshow(binary_MF_FDOG_closed, []), title('Final Vessel Map MF-FDOG');
subplot(3,2,6), imshow(GT, []), title('Ground Truth Vessel Map');

ground_truth(binary_MF_FDOG_closed)

function f = gaussian_matched_filter(s, L)
    
    % range for x and y
    t = 3; 
    x = -t*s:1:t*s;
    y = -L/2:1:L/2;

    %Calculate the normalization constant m 
    m = (1/(sqrt(2*pi)*s)) * trapz(exp(-x.^2/(2*s^2))) / (2*t*s);
    
    % 2D Gaussian MF function
    [X, Y] = meshgrid(x, y);
    f = (1/(sqrt(2*pi)*s)) * exp(-X.^2/(2*s^2)) - m;
end

function f = fdog_filter(s)
    % Create a range for x
    t = 3; 
    x = -t*s:1:t*s;
    
    % Derivative of Gaussian function
    f = -(x/(sqrt(2*pi)*s^3)) .* exp(-x.^2/(2*s^2));
end

function f = ground_truth(binary_MF_FDOG_closed)
    GT = imread('21_groundtruth.gif');
    GT = GT(:,:,1); % In case the ground truth is RGB, take one channel 
    % Check if the ground truth image needs thresholding
    if ~islogical(GT)
        GT_binary = GT > 128; % Thresholding condition if GT is not already binary
    else
        GT_binary = GT;
    end
    
    % Ensure the binary image is logical
    GT_binary = logical(GT_binary);
    
    
    % Compare final vessel map with the ground truth
    TP = sum(sum(GT_binary & binary_MF_FDOG_closed)); % True Positives
    TN = sum(sum(~GT_binary & ~binary_MF_FDOG_closed)); % True Negatives
    FP = sum(sum(~GT_binary & binary_MF_FDOG_closed)); % False Positives
    FN = sum(sum(GT_binary & ~binary_MF_FDOG_closed)); % False Negatives
    
    % Calculate performance metrics
    Accuracy = (TP + TN) / (TP + FP + FN + TN);
    Sensitivity = TP / (TP + FN); % Recall
    Specificity = TN / (TN + FP);
    Precision = TP / (TP + FP);
    F1 = 2 * ((Precision * Sensitivity) / (Precision + Sensitivity));
    
    % Display the results in a table
    resultsTable = table(Accuracy, Sensitivity, Specificity, Precision, F1)
     f = figure('Name', 'Performance Metrics', 'NumberTitle', 'off');
    
    % Create a uitable in the figure window to display the results table
    t = uitable(f, 'Data', resultsTable{:,:}, 'ColumnName', resultsTable.Properties.VariableNames, ...
        'RowName', [], 'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
    % Display the table
    disp(resultsTable);
end
