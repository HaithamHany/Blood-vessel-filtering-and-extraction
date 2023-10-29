I = imread('Vessels-test.png');
if size(I, 3) == 3
    I = rgb2gray(I); % Convert to grayscale if the image is RGB
end
I = double(I);

s = 0.5;
L = 1; 
filter = gaussian_matched_filter(s, L);

% Apply  filter
response = conv2(I, filter, 'same');

figure;
subplot(1,2,1), imshow(I, []), title('Original Image');
subplot(1,2,2), imshow(response, []), title('Matched Filter');

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