import cv2
import numpy as np
import matplotlib.pyplot as plt

'''Move target image at the position in the frame where the source is.'''
def move_target(fg_image, bg_image):
    
    # Using the foreground image, check where the object is present.
    target_bw = cv2.threshold(cv2.cvtColor(fg_image, cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)[1] 
    # select the points where the object is present
    pts_target = np.where(target_bw != 255)

    # find the boundary box points around the object
    bbox_target = np.min(pts_target[0]), np.max(pts_target[0]), np.min(pts_target[1]), np.max(pts_target[1])

    minx = np.min(bbox_target[0])
    maxx = np.max(bbox_target[0])
    miny = np.min(bbox_target[1])
    maxy = np.max(bbox_target[1])

    height = maxy- miny
    width = maxx-minx
    print(bbox_target)
    print('width',width)
    print('height',height)
    print('minx',minx)
    print('maxx',maxx)
    print('miny',miny)
    print('maxy',maxy)
    canvas = bg_image
    mask_h, mask_w, _ = bg_image.shape
    mask = np.zeros((mask_h, mask_w))
    
    # Draw mask
    k=10
    # vertical line
    mask[bbox_target[0]-k:bbox_target[1]+k+1, bbox_target[2]-k-5:bbox_target[2]+k-5] = 255
    mask[bbox_target[0]-k:bbox_target[0]+k+1, bbox_target[2]-k-5:bbox_target[3]+k+1] = 255
    mask[bbox_target[1]-k:bbox_target[1]+k+1, bbox_target[2]-k-5:bbox_target[3]+k+5] = 255
    mask[bbox_target[0]-k:bbox_target[1]+k+1, bbox_target[3]-k+5:bbox_target[3]+k+5] = 255
    cv2.imwrite('./mask/mask_temp.png', mask)

    target = bg_image[bbox_target[0]:bbox_target[1]+15, bbox_target[3]:2*bbox_target[3]-bbox_target[2]+15, :]

    target_4 = bg_image[maxx:maxx+ width+5, maxy:maxy+height+5, :]
    print('Target points: ', bbox_target[3], ', ', 2*bbox_target[3]-bbox_target[2])
    cv2.imwrite('./input/mask_temp_target.png', target_4)
    target_h, target_w, _ = target.shape
    print("Height: ", target_h)
    print("Width: ", target_w)
    cv2.imwrite('./input/mask_temp_canvas.png', canvas)
    for i, row in enumerate(range(minx, maxx)):
        for j, col in enumerate(range(miny, maxy)):
                try:
                    canvas[row, col, :] = target_4[i, j, :]
                except:
                    canvas[row, col, :] = target_4[i, j, :]

    cv2.imwrite('./input/mask_temp.png', canvas)
    
    
# Use a segmented background and foreground.
fg_image = 'foreground image'
bg_image = 'background image'

if __name__ == "__main__":
    move_target(cv2.imread(fg_image), cv2.imread(bg_image))
