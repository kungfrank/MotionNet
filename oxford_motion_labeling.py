import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
from image import load_image

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
def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

def onrelease(event):
   print('%s release: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
         ('double' if event.dblclick else 'single', event.button,
          event.x, event.y, event.xdata, event.ydata))

def onkey(event):
  global idx, gt_dict, pixel_radar_map_, ax, paint_size
  print('you pressed', event.key, event.x, event.y, event.xdata, event.ydata)
  if event.key == 'd':
    gt_dict['gt_moving'] = pixel_moving_map_.astype(np.bool)
    np.save(gt_file_path, arr=gt_dict)
    print('[save]',gt_file_path)
    idx += 1
    #plt.close()
  if event.key == 'a':
    gt_dict['gt_moving'] = pixel_moving_map_.astype(np.bool)
    np.save(gt_file_path, arr=gt_dict)
    print('[save]',gt_file_path)
    idx -= 1
    #plt.close()
  if event.key == '1':
    xdata = int(event.xdata)
    ydata = int(event.ydata)
    for i in range(-paint_size, paint_size+1):
      for j in range(-paint_size, paint_size+1):
        #print(i,j)
        x = xdata+i
        y = ydata+j
        if pixel_radar_map_[y,x] == 1:
          pixel_moving_map_[y,x] = 1
  if event.key == '2':
    xdata = int(event.xdata)
    ydata = int(event.ydata)
    for i in range(-paint_size, paint_size+1):
      for j in range(-paint_size, paint_size+1):
        #print(i,j)
        x = xdata+i
        y = ydata+j
        if pixel_radar_map_[y,x] == 1:
          pixel_moving_map_[y,x] = 0
  if event.key == ' ':
    ax.clear()
    print('[clear axis]')
  if event.key == 'z':
    paint_size=0
  if event.key == 'x':
    paint_size=1
  if event.key == 'c':
    paint_size=2
  if event.key == 'v':
    paint_size=3
  if event.key == 'b':
    paint_size=4

dataset_root = '/mnt/Disk1/training_data_mo_seg_0.05_no_mask_labeled/2019-01-10-11-46-21-radar-oxford-10k' #'2019-01-10-12-32-52-radar-oxford-10k'
filename = '2019-01-10-11-46-21-radar-oxford-10k' #'2019-01-10-12-32-52-radar-oxford-10k'

seq_dirs = [os.path.join(dataset_root, f)for f in os.listdir(dataset_root)
                 if os.path.isfile(os.path.join(dataset_root, f))]
seq_dirs.sort(key=file_to_id)

print('len(seq_dirs):', len(seq_dirs))

path = '/mnt/Disk2/Oxford/' + filename + '/'
path_cam = '/mnt/Disk2/Oxford_cam/' + filename + '/'
# Load radar ts
radar_folder = path+'radar/'
timestamps_path = path+'radar.timestamps'
radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
stereo_folder = path_cam+'stereo/centre/'
stereo_timestamps_path = path_cam+'stereo.timestamps'
stereo_timestamps = np.loadtxt(stereo_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
mono_rear_folder = path_cam+'mono_rear/'
mono_rear_timestamps_path = path_cam+'mono_rear.timestamps'
mono_rear_timestamps = np.loadtxt(mono_rear_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
mono_left_folder = path_cam+'mono_left/'
mono_left_timestamps_path = path_cam+'mono_left.timestamps'
mono_left_timestamps = np.loadtxt(mono_left_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)
mono_right_folder = path_cam+'mono_right/'
mono_right_timestamps_path = path_cam+'mono_right.timestamps'
mono_right_timestamps = np.loadtxt(mono_right_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)


fig, ax = plt.subplots(1, 1, figsize=(6, 6))
#fig_cam_f, ax_cam_f = plt.subplots(1, 1, figsize=(6, 6))
#fig_cam_l, ax_cam_l = plt.subplots(1, 1, figsize=(6, 6))
#fig_cam_r, ax_cam_r = plt.subplots(1, 1, figsize=(6, 6))
#fig_cam_b, ax_cam_b = plt.subplots(1, 1, figsize=(6, 6))
idx = 250#2100

old_idx = -1
paint_size=0
while idx>=0:
  dims = np.array([256, 256])
  if old_idx != idx:
    gt_file_path = seq_dirs[idx]
    gt_data_handle = np.load(gt_file_path, allow_pickle=True)
    gt_dict = gt_data_handle.item()

    radar_idx = file_to_id(gt_file_path)
    radar_timestamp = radar_timestamps[int(radar_idx)]
    stereo_idx, stereo_timestamp = get_sync(radar_timestamp, stereo_timestamps)
    mono_rear_idx, mono_rear_timestamp = get_sync(radar_timestamp, mono_rear_timestamps)
    mono_left_idx, mono_left_timestamp = get_sync(radar_timestamp, mono_left_timestamps)
    mono_right_idx, mono_right_timestamp = get_sync(radar_timestamp, mono_right_timestamps)

    filename = stereo_folder + str(stereo_timestamp) + '.png'
    print(filename)
    stereo_image_data = load_image(filename, model=None)
    stereo_image_data = cv2.cvtColor(stereo_image_data, cv2.COLOR_RGB2BGR)

    filename = mono_rear_folder + str(mono_rear_timestamp) + '.png'
    rear_image_data = load_image(filename, model=None)
    rear_image_data = cv2.cvtColor(rear_image_data, cv2.COLOR_RGB2BGR)

    filename = mono_left_folder + str(mono_left_timestamp) + '.png'
    left_image_data = load_image(filename, model=None)
    left_image_data = cv2.cvtColor(left_image_data, cv2.COLOR_RGB2BGR)

    filename = mono_right_folder + str(mono_right_timestamp) + '.png'
    right_image_data = load_image(filename, model=None)
    right_image_data = cv2.cvtColor(right_image_data, cv2.COLOR_RGB2BGR)

    cv2.namedWindow('left_image_data',cv2.WINDOW_NORMAL)
    cv2.imshow('left_image_data', left_image_data)
    cv2.namedWindow('right_image_data',cv2.WINDOW_NORMAL)
    cv2.imshow('right_image_data', right_image_data)
    cv2.namedWindow('rear_image_data',cv2.WINDOW_NORMAL)
    cv2.imshow('rear_image_data', rear_image_data)
    cv2.namedWindow('stereo_image_data',cv2.WINDOW_NORMAL)
    cv2.imshow('stereo_image_data', stereo_image_data)
    cv2.waitKey(250)

    raw_radars = list()
    for j in range(2):
      raw_radars.append(np.expand_dims(gt_dict['raw_radar_' + str(j)], axis=2))
    raw_radars = np.stack(raw_radars, 0).astype(np.float32)

    raw_radar_curr = raw_radars[0].squeeze()
    radar_thres = np.copy(raw_radar_curr)
    radar_thres[radar_thres>=0.08]=1
    radar_thres[radar_thres<0.08]=0

    raw_radar_curr = raw_radar_curr * radar_thres
    viz_raw_radar = np.stack((raw_radar_curr,raw_radar_curr,raw_radar_curr), axis=2)*2


    pixel_radar_map_ = gt_dict['gt_radar']
    pixel_radar_map_[pixel_radar_map_ > 0] = 1
    pixel_moving_map_ = gt_dict['gt_moving']
    pixel_moving_map_[pixel_moving_map_ > 0] = 1

    pixel_radar_map_ = pixel_radar_map_ * radar_thres

  viz_motion = np.zeros((256,256,3))
  viz_motion[:,:,0] = pixel_moving_map_ * pixel_radar_map_
  viz_motion[:,:,2] = np.logical_not(pixel_moving_map_) * pixel_radar_map_

  ax.clear()
  plt.tight_layout()
  ax.axis('off')
  ax.set_aspect('equal')
  ax.title.set_text(idx)
  ax.imshow((viz_motion+viz_raw_radar)/2.)
  #ax.imshow(pixel_moving_map_)

  cid = fig.canvas.mpl_connect('button_press_event', onclick)
  #cid = fig.canvas.mpl_connect('button_release_event', onrelease)
  cid = fig.canvas.mpl_connect('key_press_event', onkey)

  old_idx = idx
  mng = plt.get_current_fig_manager()
  mng.resize(*mng.window.maxsize())
  while plt.waitforbuttonpress() != True:
    x=0
plt.show()
