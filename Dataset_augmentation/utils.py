import cv2
import numpy as np
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance

def add_noise(image, noise_type, **kwargs):
    if noise_type == 'gaussian':
        mean = kwargs['mean']
        scale = kwargs['scale']
        gauss = np.random.normal(mean, scale, image.shape)*255
        # print(np.max(gauss), np.min(gauss))
        # print(np.max(image), np.min(image))
        output = image + gauss

    elif noise_type == 'saltpepper':
        # high = image.shape[0]*image.shape[1]*image.shape[2]
        salt = np.random.randint(2, size=(image.shape[0], image.shape[1]))
        print(salt)
        output = image.copy()
        output[salt==1] = 255

    return output

def flip_image(image, flip_dir='horizontal'):
    if flip_dir.lower() == 'vertical':
        rot_image = image[:,::-1,:]
        return rot_image[::-1,:,:]
        # return np.flipud(image)
    else:
        # return np.fliplr(image)
        return image[:,::-1,:]

def scale_image(image, scale=1, multichannel=True):
    if scale == 1:
        return image 

    image = Image.fromarray(image)
    ih, iw = image.size
    newh, neww = round(ih * scale), round(iw * scale)    
    resize = torchvision.transforms.Resize(size = (neww, newh))
    image = resize(image)

    if scale>1:
        x, y, h, w = torchvision.transforms.RandomCrop.get_params(image, output_size = (iw, ih))
        scaled_image = TF.crop(image, x, y, h, w)
    else:
        # image = np.asarray(image)
        scaled_image = Image.new('RGB', (ih, iw))
        scaled_image.paste(image, (round((neww-iw)/2), round((newh-ih)/2)))

    return np.asarray(scaled_image)

def rotate_image(image, rotangle=45):
    h, w, _ = image.shape
    M = cv2.getRotationMatrix2D((h/2, w/2), rotangle, 1)
    rotated_image = cv2.warpAffine(image, M, (h, w))
    return rotated_image

def color_augment(image, type='brightness', **kwargs):
    '''
    https://mxnet.incubator.apache.org/versions/master/tutorials/python/types_of_data_augmentation.html
    '''
    if type.lower() == 'brightness':
        val = kwargs['brightness']
        if val < 0:
            raise ValueError('Brightness value should not be negative.')
        image[image < 255-val] += val
    elif type.lower() == 'hue':
        val = kwargs['hue']
        hsvim = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsvim[..., 0] = val
        image = cv2.cvtColor(hsvim, cv2.COLOR_HSV2BGR)
    elif type.lower() == 'contrast':
        '''
        Reference: https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
        Global contrast
        '''
        val = kwargs['contrast']
        image = np.asarray(ImageEnhance.Contrast(Image.fromarray(image)).enhance(val))
    elif type.lower() == 'grayscale':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def translate_image(image, tx = -10, ty = 10):
    h, w, _ = image.shape
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
    return translated_image
