img=imread('test.jpg');
A=img(:,:,1);
%size(A)

ne = 3; %define the neighborhood
idx = floor(ne/2);

modified_img = zeros(size(A));

% grayscale image
for i=1:size(A, 1)
    for j=1:size(A, 2)
        count = 0;
        for k1=-idx:idx
            for k2=-idx:idx
                if (i+k1)>0 && (j+k2)>0 && (i+k1)<=size(A, 1) && (j+k2)<=size(A,2)
                    modified_img(i, j) = modified_img(i, j)+ A(i+k1, j+k2);
                    count = count + 1;            
                end
            end
        end
        modified_img(i, j) = modified_img(i, j)/count;
    end
end

meanImage = imfilter(A, ones(3)/9);

meanFilter = fspecial('average', [3 3]);
toShow = imfilter(A, meanFilter);

% takes a lot of compute time
% RGB image
mod_img = zeros(size(img));

for i=1:size(img, 1)
    for j=1:size(img, 2)
        for l=1:size(img, 3)
            count = 0;
            for k1=-idx:idx
                for k2=-idx:idx
                    if (i+k1)>0 && (j+k2)>0 && (i+k1)<=size(img, 1) && (j+k2)<=size(img,2)
                        mod_img(i, j, l) = mod_img(i, j, l)+ img(i+k1, j+k2, l);
                        count = count + 1;
                    end
                end
            end
            mod_img(i, j, l) = mod_img(i, j, l)/count;
        end
    end
end

figure()
subplot(2, 2, 1);
imshow(img, []);
title('Original image');
subplot(2, 2, 2);
imshow(mod_img, []);
title('RGB Averaged image');
subplot(2, 2, 3);
imshow(modified_img, []);
title('Averaged pixels');
subplot(2, 2, 4);
imshow(toShow, []);
title('Using command');
size(toShow)
