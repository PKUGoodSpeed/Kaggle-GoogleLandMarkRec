import multiprocessing
import os
from io import BytesIO
from urllib import request
import pandas as pd
import re
import tqdm
from PIL import Image
from multiproc import MyPool

# set files and dir
DATA_FRAME, OUT_DIR = pd.read_csv('../input/train.csv'), '../data/train'
# DATA_FRAME, OUT_DIR = pd.read_csv('../input/test.csv'), '../input/test'  # test data

# preferences
TARGET_SIZE = 128  # image resolution to be stored
IMG_QUALITY = 90  # JPG quality
NUM_WORKERS = 4  # Num of CPUs

def download_image(key_url):
    key, url = key_url
    filename = os.path.join(OUT_DIR, '{}.jpg'.format(key))
    
    if os.path.exists(filename):
        print('Image {} already exists. Skipping download.'.format(filename))
        return 0
    
    try:
        response = request.urlopen(url)
        image_data = response.read()
    except:
        print('Warning: Could not download image {} from {}'.format(key, url))
        return 1
    
    try:
        pil_image = Image.open(BytesIO(image_data))
    except:
        print('Warning: Failed to parse image {}'.format(key))
        return 1
    
    try:
        pil_image_rgb = pil_image.convert('RGB')
    except:
        print('Warning: Failed to convert image {} to RGB'.format(key))
        return 1
    
    try:
        pil_image_resize = pil_image_rgb.resize((TARGET_SIZE, TARGET_SIZE))
    except:
        print('Warning: Failed to resize image {}'.format(key))
        return 1
    
    try:
        pil_image_resize.save(filename, format='JPEG', quality=IMG_QUALITY)
    except:
        print('Warning: Failed to save image {}'.format(filename))
        return 1
    
    return 0


if __name__ == '__main__':
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    kargs = [(k, u) for k, u in zip(DATA_FRAME.id.tolist(), DATA_FRAME.url.tolist())]
    print(len(kargs))
    pool = MyPool(NUM_WORKERS)
    pool.map(download_image, kargs)
