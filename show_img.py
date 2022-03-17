import os
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from image import load_image

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.png')]
  idx = int(float(idx))
  return idx
def UnixTimeToSec(unix_timestamp):
    #print('Unix timestamp:')
    #print(str(unix_timestamp))
    time = datetime.fromtimestamp(unix_timestamp / 1000000)
    s = unix_timestamp % 1000000
    sec_timestamp = time.hour*3600 + time.minute*60 + time.second + (float(s)/1000000)
    #print("timestamp: ")
    #print(sec_timestamp)
    return sec_timestamp
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
def poler_to_cart(raw_example_data, cart_resolution, cart_pixel_width, interpolate_crossover):
  ##########################################################################################
  radar_resolution = np.array([0.0432], np.float32)
  encoder_size = 5600

  timestamps = raw_example_data[:, :8].copy().view(np.int64)
  azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
  valid = raw_example_data[:, 10:11] == 255
  fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255. # (400, 3768, 1)
  ##########################################################################################
  if (cart_pixel_width % 2) == 0:
      cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
  else:
      cart_min_range = cart_pixel_width // 2 * cart_resolution
  coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)
  Y, X = np.meshgrid(coords, -coords)
  sample_range = np.sqrt(Y * Y + X * X)
  sample_angle = np.arctan2(Y, X)
  sample_angle += (sample_angle < 0).astype(np.float32) * 2. * np.pi

  # Interpolate Radar Data Coordinates
  azimuth_step = azimuths[1] - azimuths[0]
  sample_u = (sample_range - radar_resolution / 2) / radar_resolution
  sample_v = (sample_angle - azimuths[0]) / azimuth_step

  # We clip the sample points to the minimum sensor reading range so that we
  # do not have undefined results in the centre of the image. In practice
  # this region is simply undefined.
  sample_u[sample_u < 0] = 0

  if interpolate_crossover:
      fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0) # (402, 3768, 1)
      sample_v = sample_v + 1

  polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
  cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
  return cart_img


filename = '2019-01-10-12-32-52-radar-oxford-10k' #'2019-01-10-11-46-21-radar-oxford-10k' #'2019-01-10-12-32-52-radar-oxford-10k'#'2019-01-10-12-32-52-radar-oxford-10k'

data_path = '/mnt/Disk1/viz_result_3fig/2021-04-13_15-32-07_ep50_test/occupied' # '/mnt/Disk1/viz_result/radar2lidar/2021-03-11_12-57-31_single_plot'
data_dirs = [os.path.join(data_path, f)for f in os.listdir(data_path)
                 if os.path.isfile(os.path.join(data_path, f))]
data_dirs.sort(key=file_to_id)
print('len(data_dirs):',len(data_dirs))

data_path2 = '/mnt/Disk1/viz_result_3fig/2021-04-09_16-28-24_test/occupied'
data_dirs2 = [os.path.join(data_path2, f)for f in os.listdir(data_path2)
                 if os.path.isfile(os.path.join(data_path2, f))]
data_dirs2.sort(key=file_to_id)
print('len(data_dirs2):',len(data_dirs2))

win1_name = 'w/ power_thres_gt'
win2_name = 'w/o power_thres_gt'

cart_resolution = 0.25
cart_pixel_width = 256
interpolate_crossover = True

init = True

path = '/mnt/Disk2/Oxford_radar/' + filename + '/'
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

cv2.namedWindow('left_image_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('left_image_data', 450, 450)
cv2.namedWindow('right_image_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('right_image_data', 450, 450)
cv2.namedWindow('rear_image_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('rear_image_data', 450, 450)
cv2.namedWindow('stereo_image_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('stereo_image_data', 450, 450)
cv2.namedWindow('radar_image_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('radar_image_data', 450, 450)
cv2.namedWindow('radar_raw_data',cv2.WINDOW_NORMAL)
cv2.resizeWindow('radar_raw_data', 450, 450)

cv2.namedWindow(win1_name,cv2.WINDOW_NORMAL)
cv2.namedWindow(win2_name,cv2.WINDOW_NORMAL)

fig, ax = plt.subplots(1, 1, figsize=(7, 7))


idx = 0
auto_playing_mode = True

while 1:
  img_path = data_dirs[idx]
  img_path2 = data_dirs2[idx]
  print(img_path)
  print(img_path2)
  radar_idx = file_to_id(img_path)
  if radar_idx < 554: #330
    idx+=1
    continue
  img = cv2.imread(img_path)
  img2 = cv2.imread(img_path2)

  radar_timestamp = radar_timestamps[int(radar_idx)]
  stereo_idx, stereo_timestamp = get_sync(radar_timestamp, stereo_timestamps)
  mono_rear_idx, mono_rear_timestamp = get_sync(radar_timestamp, mono_rear_timestamps)
  mono_left_idx, mono_left_timestamp = get_sync(radar_timestamp, mono_left_timestamps)
  mono_right_idx, mono_right_timestamp = get_sync(radar_timestamp, mono_right_timestamps)

  filename = radar_folder + str(radar_timestamp) + '.png'
  if not os.path.isfile(filename):
      print("Could not find cam example: {}".format(filename))
  radar_data = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

  #cv2.imshow('radar_raw_data', radar_data)
  #cart_img = poler_to_cart(radar_data, cart_resolution, cart_pixel_width, interpolate_crossover)

  #cart_img = cv2.cvtColor(cart_img, cv2.COLOR_GRAY2BGR)
  #cart_img = np.uint8(cart_img*255)
  #cv2.imshow('radar_image_data', cart_img)


  filename = stereo_folder + str(stereo_timestamp) + '.png'
  if not os.path.isfile(filename):
      print("Could not find cam example: {}".format(filename))
  stereo_image_data = load_image(filename, model=None)
  stereo_image_data = cv2.cvtColor(stereo_image_data, cv2.COLOR_RGB2BGR)

  filename = mono_rear_folder + str(mono_rear_timestamp) + '.png'
  if not os.path.isfile(filename):
      print("Could not find cam example: {}".format(filename))
  rear_image_data = load_image(filename, model=None)
  rear_image_data = cv2.cvtColor(rear_image_data, cv2.COLOR_RGB2BGR)

  filename = mono_left_folder + str(mono_left_timestamp) + '.png'
  if not os.path.isfile(filename):
      print("Could not find cam example: {}".format(filename))
  left_image_data = load_image(filename, model=None)
  left_image_data = cv2.cvtColor(left_image_data, cv2.COLOR_RGB2BGR)

  filename = mono_right_folder + str(mono_right_timestamp) + '.png'
  if not os.path.isfile(filename):
      print("Could not find cam example: {}".format(filename))
  right_image_data = load_image(filename, model=None)
  right_image_data = cv2.cvtColor(right_image_data, cv2.COLOR_RGB2BGR)

  cv2.imshow('left_image_data', left_image_data)
  cv2.imshow('right_image_data', right_image_data)
  cv2.imshow('rear_image_data', rear_image_data)
  cv2.imshow('stereo_image_data', stereo_image_data)

  cv2.imshow(win1_name, img)
  cv2.imshow(win2_name, img2)

  if init:
    cv2.waitKey(0)
    init = False
  else:
    if auto_playing_mode:
      idx += 1
      key = cv2.waitKey(10)
      if key == 32: # space
        auto_playing_mode = not auto_playing_mode
    else:
      key = cv2.waitKey(0)
      if key == 100: # d
        idx += 1
      if key == 97: # a
        idx -= 1
      if idx < 0:
        idx=0
      if key == 32: # space
        auto_playing_mode = not auto_playing_mode
