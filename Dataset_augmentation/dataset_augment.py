''' 
    Scipt for Dataset augmentation
'''
import cv2
from utils import *
import matplotlib.pyplot as plt

img = cv2.imread('cat_image.png')

'''
1 - Add random noise to the image  
'''
noisy_image = add_noise(img, 'gaussian', mean=0.0, scale=0.1**0.5)

'''
2 - Random Horizontal and Vertical flip 
'''
flipped_image = flip_image(img, 'horizontal')
cv2.imwrite("./flip_im.jpg", flipped_image)

'''
3 - Randomly scale the images 
'''
scaled_image = scale_image(img, scale=1.5, multichannel=True)
cv2.imwrite("./scale_im.jpg", scaled_image)

'''
4 - Rotating an image
'''
rotated_image = rotate_image(img, rotangle=45)
cv2.imwrite("./rot_im.jpg", rotated_image)

'''
5 - Color augmentation of the image
'''
# mod_image = color_augment(img, type='brightness', brightness=50)
# mod_image = color_augment(img, type='hue', hue=50)
# mod_image = color_augment(img, type='contrast', contrast=200)
mod_image = color_augment(img, type='grayscale')
cv2.imwrite("./color_aug.jpg", mod_image)

'''
6 - Random translation of the image (left, right, up and down)
params:
tx = move tx units towards the right
ty = move ty units down
'''
translated_image = translate_image(img, tx = -100, ty = 20)
