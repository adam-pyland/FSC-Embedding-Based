"""move 100 files from '/work/Sultan/data/nopickup/' both from images/val folder and labels/val folder
to  '/work/Sultan/data/nopickup/ images/test and labels/test folder'"""

import os, glob
import shutil
from random import sample as sample


src = '/work/Sultan/data/nopickup/images/val_yolo/'
dst = '/work/Sultan/data/nopickup/images/test_yolo/'
if not os.path.exists(dst):
    os.makedirs(dst)
if not os.path.exists(dst.replace('images','labels')):
    os.makedirs(dst.replace('images','labels'))
num = 99
img_list = glob.glob(os.path.join(src, '*.jpg'))
samp_files = sample(img_list, num)
for file in samp_files:
    shutil.move(file, dst)
    shutil.move(file.replace('images','labels').replace('.jpg','.txt'), dst.replace('images','labels'))





