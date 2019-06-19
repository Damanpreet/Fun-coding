image = imread('test.jpg');
rot_image = imrotate(image, 45);
rot_image_90 = imrotate(image, 90);

figure();
subplot(2, 2, 1);
imshow(image)
title('Original image')
subplot(2, 2, 2);
imshow(rot_image)
title('Rotated image')
subplot(2, 2, 3);
imshow(rot_image_90);
title('Rotated image - 90degrees');


