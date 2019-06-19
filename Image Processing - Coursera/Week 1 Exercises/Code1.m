intensity_value = input('Intensity value: ');
scale = 256/intensity_value;
image = imread('test.jpg');

A1 = image(:,:,1); %grayscale 
A = image;
size(A)

image_reduced = zeros(size(A));
for i=1:size(A, 1)
    for j=1:size(A, 2)
        for k=1:size(A, 3)
            % get the pixel value
            image_reduced(i, j, k) = floor(A(i, j, k)/scale) * scale;    
        end
    end
end

image_reduced_gray = zeros(size(A1));
for i=1:size(A1, 1)
    for j=1:size(A1, 2)
        % get the pixel value
        image_reduced_gray(i, j) = floor(double(A1(i, j))/double(scale)) * scale;
    end
end

%Plot the figure
figure()
subplot(2, 2, 1)
imshow(image, [])
title('Original image')
subplot(2, 2, 2)
imshow(image_reduced, [])
title('Reduced image')
subplot(2, 2, 3)
imshow(image_reduced_gray, [])
title('Reduced image grayscale')
