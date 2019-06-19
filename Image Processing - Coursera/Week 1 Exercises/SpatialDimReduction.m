img = imread('test.jpg');
img = img(:,:,1);
dim = 10;

im_size = size(img);
avg_img = zeros(im_size(1)/dim, floor(im_size(2)/dim));

for i=1:im_size(1)/dim
    for j=1:im_size(2)/dim
        if ((i*dim) <= im_size(1)) && ((j*dim) <= im_size(2)) 
            img_patch = img(i*dim-dim+1:i*dim, j*dim-dim+1:j*dim);
            avg_img(i, j)=mean(img_patch(:));     
        end
    end
end
 
figure()
subplot(1, 2, 1);
imshow(avg_img, [])
title('Averaged image')
subplot(1, 2, 2);
imshow(img, [])
title('Original image')