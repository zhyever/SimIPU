import mmcv
import pickle
import os
import numpy as np
from shutil import copyfile

root = '/mnt/10-5-108-187/lizhenyu1/kitti_det/public_datalist_14/'
sources = ['testing/velodyne_reduced','training/velodyne_reduced']

target = 'traintest/velodyne_reduced'
target = os.path.join(root, target) 
if not os.path.exists(target):
    os.makedirs(target)

arr = []

source = sources[1]
source = os.path.join(root, source) 
base = len(os.listdir(source))
for f in os.listdir(source):
    source_file = os.path.join(source, f)
    destination_file = os.path.join(target, f)
    arr.append(destination_file)
    copyfile(source_file, destination_file)

source = sources[0]
source = os.path.join(root, source) 
for f in os.listdir(source):
    source_file = os.path.join(source, f)
    idx = int(f.split('.')[0])
    idx += base
    destination_file = f'{idx:06d}.bin'.format(idx=idx)
    destination_file = os.path.join(target, destination_file)
    arr.append(destination_file)
    # print(f, destination_file)
    copyfile(source_file, destination_file)

print(base)
print(len(np.unique(arr)))