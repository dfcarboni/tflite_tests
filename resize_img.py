import os
import argparse
import numpy as np
import glob
from PIL import Image
import matplotlib.pyplot as plt

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path of image',
                    required=True)
parser.add_argument('--img_size', help='new image size',
                    required=True, nargs=2)
parser.add_argument('--crop', help='crop image before resizing',
                    action='store_true')

args = parser.parse_args()

IMAGE_PATH = args.image
IMG_SIZE = args.img_size
CROP= args.crop

# IMG_SIZE= int(IMG_SIZE)
# IMAGE_PATH= 'buva.png'
# CROP= 1
# IMG_SIZE= (300,300)


def center_crop(image, new_w= None, new_h= None):
    # PIL image
    w,h= image.size
    
    if not new_w:
        new_w = w
    if not new_h:
        new_h= h
    
    left = (w - new_w)/2
    top = (h - new_h)/2
    right = (w + new_w)/2
    bottom = (h + new_h)/2
    
    image = image.crop((left, top, right, bottom))

    return image


image= Image.open(IMAGE_PATH)

w,h= image.size
sq1= min(w,h)

if CROP:
    if w > sq1:
        image= center_crop(image, new_w= sq1)
        
    if h > sq1:
        image= center_crop(image, new_h= sq1)
    else:
        pass

new_w,new_h= int(IMG_SIZE[0]), int(IMG_SIZE[1])
# print(new_w, new_h)
image= image.resize((new_w,new_h), Image.ANTIALIAS)

# plt.imshow(image)
if CROP:
    image.save(f'{IMAGE_PATH[:-4]}_{new_w}x{new_h}crop{IMAGE_PATH[-4:]}')
else:
    image.save(f'{IMAGE_PATH[:-4]}_{new_w}x{new_h}{IMAGE_PATH[-4:]}')