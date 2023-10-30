I = imread('Vessle-Test02.png');
if size(I, 3) == 3
    I = rgb2gray(I); % Convert to grayscale if the image is RGB
end

% Contrast Enhancement using Adaptive Histogram Equalization
I_uint8 = im2uint8(I); % Convert to uint8 for adapthisteq
I_enhanced_uint8 = adapthisteq(I_uint8,'ClipLimit',0.02); 
I = im2double(I_enhanced_uint8); % Convert back to double for further processing

s = 0.5;
L = 0.5; 
filter = gaussian_matched_filter(s, L);

% Apply  filter
response = conv2(I, filter, 'same');

% Background subtraction
se = strel('disk', 15);
background = imopen(response, se);
vessel_enhanced = response - background;

% Display
figure;
subplot(1,2,1), imshow(I, []), title('Original Image');
subplot(1,2,2), imshow(vessel_enhanced, []), title('Vessel Enhanced');

function f = gaussian_matched_filter(s, L)
    % s is the scale of the filter
    % L is the length of the neighborhood along the y-axis to smooth noise
    
    % range for x and y
    t = 3; 
    x = -t*s:1:t*s;
    y = -L/2:1:L/2;
    
    % Calculate the normalization constant m
    m = (1/(sqrt(2*pi)*s)) * trapz(exp(-x.^2/(2*s^2))) / (2*t*s);
    
    % 2D Gaussian MF function
    [X, Y] = meshgrid(x, y);
    f = (1/(sqrt(2*pi)*s)) * exp(-X.^2/(2*s^2)) - m;
end