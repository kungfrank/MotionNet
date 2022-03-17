import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
from image import load_image
from sklearn.cluster import DBSCAN
from skimage import color
import time

global idx, gt_dict, pixel_radar_map_, ax, paint_size

def find(s, ch):
  return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
  idx = int(idx)
  return idx

def get_sync(t, timestamps):
    """get the closest id given the timestamp

    :param t: timestamp in seconds
    :type t: float
    :param all_timestamps: a list with all timestamps
    :type all_timestamps: np.array
    :param time_offset: offset in case there is some unsynchronoised sensor, defaults to 0.0
    :type time_offset: float, optional
    :return: the closest id
    :rtype: int
    """
    idx = np.argmin(np.abs(timestamps - t))
    return idx, timestamps[idx]
dataset_root = '/mnt/Disk1/training_data_local_global_vel_i3o1_with_tf_no_mask_labeled/2019-01-10-11-46-21-radar-oxford-10k' #'2019-01-10-12-32-52-radar-oxford-10k'
filename = '2019-01-10-11-46-21-radar-oxford-10k' #'2019-01-10-12-32-52-radar-oxford-10k'

seq_dirs = [os.path.join(dataset_root, f)for f in os.listdir(dataset_root)
                 if os.path.isfile(os.path.join(dataset_root, f))]
seq_dirs.sort(key=file_to_id)

print('len(seq_dirs):', len(seq_dirs))

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

idx = 0

old_idx = -1
paint_size=0
while idx>=0:
  gt_file_path = seq_dirs[idx]
  gt_data_handle = np.load(gt_file_path, allow_pickle=True)
  gt_dict = gt_data_handle.item()
  pixel_radar_map_ = gt_dict['gt_radar']
  pixel_radar_map_[pixel_radar_map_>0]=1
  pixel_moving_map_ = gt_dict['gt_moving']
  pixel_moving_map_[pixel_moving_map_>0]=1
  raw_radar = gt_dict['raw_radar_0']

  power_thres_map=np.copy(raw_radar)
  power_thres_map[power_thres_map>0.08] = 1
  power_thres_map[power_thres_map<=0.08] = 0

  pixel_radar_map_ = pixel_radar_map_ * pixel_moving_map_ # * power_thres_map
  #pixel_radar_map_ = pixel_moving_map_

  pixel_disp_global_map_ = gt_dict['gt_disp_global'] * np.expand_dims(pixel_radar_map_, axis=2) * np.expand_dims(pixel_moving_map_, axis=2)
  pixel_disp_global_norm_map_ = np.linalg.norm(pixel_disp_global_map_, axis=2)

  points = np.where(pixel_radar_map_ == 1)
  points_vel = pixel_disp_global_norm_map_[points[0],points[1]] * 3.

  if len(points[0])==0:
    print('no moving obj')
    idx+=1
    continue

  points_ = np.concatenate((points[0].reshape(-1,1), points[1].reshape(-1,1), np.zeros(points[0].shape).reshape(-1,1)), axis=1)
  #points_ = np.concatenate((points[0].reshape(-1,1), points[1].reshape(-1,1), points_vel.reshape(-1,1)), axis=1)
  print(points_.shape)

  t1 = time.time()
  clustering = DBSCAN(eps=6, min_samples=2).fit(points_)
  t2 = time.time()
  print('dbscan cost:', t2-t1)
  labels = clustering.labels_

  labels = labels.astype(np.int)
  points_ = points_.astype(np.int)
  print(labels.shape)

  res = np.zeros((256,256))
  for i in range(points_.shape[0]):
    #print(i, points_[i,0], points_[i,1], labels[i])
    res[points_[i,0], points_[i,1]] = labels[i]+1

  color_res = color.label2rgb(res,pixel_radar_map_,bg_label=0)

  fig, ax = plt.subplots(1, 3, figsize=(18, 6))
  #ax.imshow(res, cmap='rainbow')
  #ax.imshow(color_res)
  ax[0].imshow(gt_dict['gt_radar'])
  ax[1].imshow(pixel_disp_global_norm_map_)
  ax[2].imshow(color_res)
  idx+=1
  #plt.pause(0.1)
  plt.show()


