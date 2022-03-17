# python3 no ros

import argparse
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
import time
import math

from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os
pid = os.getpid()
print('pid:', pid)

log_name = '2019-01-10-12-32-52-radar-oxford-10k' # '2019-01-10-11-46-21-radar-oxford-10k' '2019-01-10-12-32-52-radar-oxford-10k'

MODE = 'test' # 'test' 'train'
ODOM_COMP = True
MASKING = True

save_directory = '/mnt/Disk1/training_data_vel/'+log_name

history_scan_num = 5

cart_resolution = 0.25
cart_pixel_width = 256
interpolate_crossover = True

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
#fig_single, ax_single = plt.subplots(1, 5, figsize=(8, 8))
plt.tight_layout()

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

def get_sync_lidar_scans(t, timestamps):
    idx = np.argmin(np.abs(timestamps - t))
    idx_list = np.arange(5)+idx
    return idx_list, timestamps[idx_list]

def polar_to_cart(raw_example_data):
    ##########################################################################################
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.
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
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    return cart_img


def load_sync_data(curr_radar_t, lidar_res_timestamps, lidar_rmg_timestamps, lidar_res_path, lidar_rmg_path):
    res_idx_list, res_ts_list = get_sync_lidar_scans(curr_radar_t, lidar_res_timestamps)
    rmg_idx_list, rmg_ts_list = get_sync_lidar_scans(curr_radar_t, lidar_rmg_timestamps)

    motion_seg_list = list()
    disp_0_list = list()
    disp_1_list = list()
    for ts in res_ts_list:
      # Load npy
      p = lidar_res_path+str(format(ts,'.7f'))+'.npy'
      res_data = np.load(p , allow_pickle=True)

      # Read dict
      disp_pred = res_data.item()['disp_pred']
      non_empty_map = res_data.item()['non_empty_map']
      motion_pred = res_data.item()['motion_pred']

      # Coordinate trnasform
      disp_pred = np.rot90(disp_pred)
      non_empty_map = np.rot90(non_empty_map)
      motion_pred = np.rot90(motion_pred)

      # Pre-process
      mo_map = motion_pred.squeeze() * non_empty_map
      vel_map = np.sqrt(np.power(disp_pred[:,:,0],2)+np.power(disp_pred[:,:,1],2)) * non_empty_map
      binary_vel_map = np.ones_like(vel_map)
      thd_mask = vel_map < 0.5 # 0.5 # !!!!!!!!!!!!!!!!! This is used in initial test, but this is too big to segment moving pedestrian
      binary_vel_map[thd_mask] = 0

      motion_seg_list.append(binary_vel_map)

      #disp_pred = np.rot90(disp_pred,2)
      disp_0_list.append(disp_pred[:,:,0])
      disp_1_list.append(disp_pred[:,:,1])


    rmg_data_list = list()
    for ts in rmg_ts_list:
      p = lidar_rmg_path+str(format(ts,'.7f'))+'.npy'
      rmg_data = np.load(p , allow_pickle=True)
      rmg_non_empty_map = rmg_data.item()['non_empty_map'].astype(np.float32)
      rmg_non_empty_map = np.flip(rmg_non_empty_map,axis=0)
      rmg_data_list.append(rmg_non_empty_map)

    return motion_seg_list, disp_0_list, disp_1_list, rmg_data_list


def img_cart_to_polar(img):
  img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
  img = np.rot90(img,-1) # rot clockwise 90 deg
  value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
  polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
  return polar_image

def img_polar_to_cart(img):
  img = np.concatenate((img[int(img.shape[0]/4):,:], img[:int(img.shape[0]/4),:]), axis=0)
  value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
  cart_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
  cart_image = cv2.resize(cart_image, (256, 256), interpolation=cv2.INTER_LINEAR)
  return cart_image

def compute_gradient_direction(array, index, return_grad=False):
  #print('compute_gradient_direction','index', index,'start')
  if index == array.shape[0]-1: ### temporary work to solve empty issue in polar radar scan !!!!
    return 1, 0

  not_converge = True
  i = 0
  grad_direct = 0
  while not_converge:
    i = i + 1
    #print('compute_gradient_direction','index', index, 'i', i)
    if index+i >= array.shape[0]:
      grad = (array[index] - array[index-i])/(i)
    elif index-i < 0:
      grad = (array[index+i] - array[index])/(i)
    else:
      grad = (array[index+i] - array[index-i])/(2*i)
    #print(grad)
    if grad != 0:
      not_converge = False
  if grad > 0:
    grad_direct = 1
  else:
    grad_direct = 0

  #print('compute_gradient_direction','index', index, grad_direct, grad)
  if return_grad == False:
    return grad_direct
  else:
    return grad_direct, grad

def find_local_max(array, index):
  #print('find_local_max start')
  #print 'id:', index
  #figure, axis = plt.subplots(1,1)
  #axis.plot(np.arange(len(array)), array)
  #axis.set(xlim=(0, 1000))
  #axis.axvline(x=index, color='k', linestyle='--')
  not_converge = True
  init = True
  init_dir = 0
  while not_converge:
    dir = compute_gradient_direction(array, index)
    if init:
      init_dir = dir
      init = False
      continue
    if dir == 1:
      index = index+1
    elif dir == 0:
      index = index-1
    if init_dir != dir:
      not_converge = False
  #axis.axvline(x=index, color='red', linestyle='-')
  #plt.show()
  #print( 'find_local_max return', index)
  return index

def generate_mask_with_local_max(list, array):
  masked_array = np.zeros(len(array))

  #print('generate_mask_with_local_max start')
  for i in list:
    #print i
    left_bound = i
    right_bound = i
    finding_left_bound = True
    finding_right_bound = True
    #print(i, 'finding_left_bound...')
    while finding_left_bound:
      left_bound = left_bound-1
      dir, grad = compute_gradient_direction(array, left_bound, return_grad=True)
      if dir == 0 or abs(grad) < 0.01:
        finding_left_bound = False
    #print(i, 'finding_right_bound...')
    while finding_right_bound:
      right_bound = right_bound+1
      dir, grad = compute_gradient_direction(array, right_bound, return_grad=True)
      if dir == 1 or abs(grad) < 0.01:
        finding_right_bound = False
    #print left_bound, right_bound
    masked_array[left_bound+1:right_bound] = 1
    #print masked_array.shape
  return masked_array


def mask_radar_by_lidar(radar_img_, lidar_img_):
    #radar_img_ = cv2.GaussianBlur(radar_img_,(1,5),0) ############## Blur !!!!!!!!!!
    result_img = radar_img_.copy()

    for i in range(lidar_img_.shape[0]): # range(lidar_img_.shape[0])
      #print( 'row num:', i)
      local_max_list = []
      id_list = np.where(lidar_img_[i,:] == 1.)[0]
      #print( 'id_list', id_list)
      for id_ in id_list:
        local_max_list.append(find_local_max(radar_img_[i,:], id_))

      ## show local maximum
      result_row = np.zeros(radar_img_.shape[1])
      one_array = np.ones(len(local_max_list))
      result_row.put(local_max_list,one_array*1.)
      result_img[i] = result_row

      ## show final mask
      masked_row = generate_mask_with_local_max(local_max_list,radar_img_[i,:])
      result_img[i] = masked_row*1.

    return result_img

def combined_lidar_scans(data_list):
  combined_polar_lidar = np.array([])
  for i in range(len(data_list)):
    polar_lidar = img_cart_to_polar(data_list[i])
    polar_lidar_clip = polar_lidar[i*52:(i+1)*52,:]
    if i == 0:
      combined_polar_lidar = polar_lidar_clip
    else:
      combined_polar_lidar = np.concatenate((combined_polar_lidar, polar_lidar_clip), axis=0)
  return combined_polar_lidar

def gen_gt_mask(rmg_data_list, radar_cart):
  combined_polar_lidar = combined_lidar_scans(rmg_data_list) # motion_seg_list rmg_data_list
  combined_polar_lidar[combined_polar_lidar<0.5] = 0
  combined_polar_lidar[combined_polar_lidar>0.5] = 1

  radar_polar = img_cart_to_polar(radar_cart)

  if MASKING == True:
    # masking #
    result_polar_radar = mask_radar_by_lidar(radar_polar, combined_polar_lidar)
  else:
    # no masking #
    result_polar_radar = combined_polar_lidar

  result_cart_radar = img_polar_to_cart(result_polar_radar)

  combined_cart_lidar = img_polar_to_cart(combined_polar_lidar)
  combined_cart_lidar[combined_cart_lidar<0.5] = 0
  combined_cart_lidar[combined_cart_lidar>0.5] = 1
  return result_cart_radar, combined_cart_lidar, result_polar_radar, combined_polar_lidar # with mask, without mask

def gen_gt_vel(disp_list):
  combined_polar_lidar = combined_lidar_scans(disp_list)
  result_cart_radar = img_polar_to_cart(combined_polar_lidar)
  return result_cart_radar

def warp_radar_by_radar_motion(src, tf_mat):
  delta_x = tf_mat[0,2]
  delta_y = tf_mat[1,2]
  delta_yaw = np.arctan(tf_mat[1,0] / tf_mat[0,0]) #### !! might have ambiguous issue between pi/2 and -pi/2
  #print(delta_x, delta_y, delta_yaw)
  cv_x = -delta_y/cart_resolution
  cv_y = delta_x/cart_resolution
  cv_theta = delta_yaw

  warp_mat = np.zeros((2,3), dtype=np.float32)
  warp_mat[0,0] = np.cos(cv_theta)
  warp_mat[0,1] = np.sin(cv_theta)
  warp_mat[0,2] = cv_x*np.cos(cv_theta) + cv_y*np.sin(cv_theta) + ( (1-np.cos(cv_theta))*src.shape[1]/2 - np.sin(cv_theta)*src.shape[0]/2 )
  warp_mat[1,0] = -np.sin(cv_theta)
  warp_mat[1,1] = np.cos(cv_theta)
  warp_mat[1,2] = cv_x*(-np.sin(cv_theta)) + cv_y*np.cos(cv_theta) + ( np.sin(cv_theta)*src.shape[1]/2 + (1-np.cos(cv_theta))*src.shape[0]/2 )
  warp_dst = cv2.warpAffine(src, warp_mat, (src.shape[1], src.shape[0]))

  return warp_dst

def load_radar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num):
  radar_cart_list = list()
  #print('radar_idx',radar_idx)
  for i in range(history_scan_num):
    #print('i',i)
    idx = radar_idx - i
    radar_filename = radar_folder + str(radar_timestamps[idx]) + '.png'
    radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
    radar_cart = polar_to_cart(radar_img).squeeze()
    tf_mat = np.identity(3).astype(np.float32)
    for j in range(i):
      j = i-1-j
      ro_idx = radar_idx-1-j
      #print('ro_idx',ro_idx)
      ro_tf = GetMatFromXYYaw(ro_data.iloc[ro_idx,2],ro_data.iloc[ro_idx,3],ro_data.iloc[ro_idx,7])
      tf_mat = tf_mat.dot(ro_tf)
    trans_radar_cart = warp_radar_by_radar_motion(radar_cart, tf_mat)
    #cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    #cv2.imshow('Image', trans_radar_cart)
    #cv2.waitKey(0)
    radar_cart_list.append(trans_radar_cart)
  return radar_cart_list

def load_radar_data_wo_odom_comp(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num):
  radar_cart_list = list()
  #print('radar_idx',radar_idx)
  for i in range(history_scan_num):
    idx = radar_idx - i
    radar_filename = radar_folder + str(radar_timestamps[idx]) + '.png'
    radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
    radar_cart = polar_to_cart(radar_img).squeeze()
    radar_cart_list.append(radar_cart)
  return radar_cart_list

def load_radar_polar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, history_scan_num):
  radar_polar_list = list()
  #print('radar_idx',radar_idx)
  for i in range(history_scan_num):
    idx = radar_idx - i
    radar_filename = radar_folder + str(radar_timestamps[idx]) + '.png'
    radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
    #radar_cart = polar_to_cart(radar_img).squeeze()
    radar_polar_list.append(radar_img.squeeze())
  return radar_polar_list

def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat

def main():

    #fig, ax = plt.subplots(2, 3, figsize=(20, 14))
    #plt.tight_layout()
    for filename in os.listdir("/mnt/Disk2/Oxford"):
      if filename != log_name: # '2019-01-10-11-46-21-radar-oxford-10k' '2019-01-10-12-32-52-radar-oxford-10k'
        continue
      print(filename)

      path = '/mnt/Disk2/Oxford/' + filename + '/'
      radar_folder = path+'radar/'
      timestamps_path = path+'radar.timestamps'
      radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

      ro_file = path+'gt/radar_odometry.csv'
      ro_data = pd.read_csv(ro_file)
      ro_timestamps = ro_data.iloc[:,8].copy()

      gps_file = path+'/gps/ins.csv'
      gps_data = pd.read_csv(gps_file)
      gps_timestamps = gps_data.iloc[:,0].copy()

      lidar_res_path = '/mnt/Disk1/oxford_res_combined_ts_from_pseudo_lidar/'+log_name+'/'
      lidar_res_timestamps_path = lidar_res_path+'pseudo_lidar.timestamps'
      lidar_res_timestamps = np.loadtxt(lidar_res_timestamps_path, delimiter=' ', usecols=[0], dtype=np.float64)
      #print(lidar_res_timestamps.shape)
      #print(lidar_res_timestamps[:10])

      lidar_rmg_path = '/mnt/Disk1/oxford_rmg_lidar_combined_ts_from_radar/'+log_name+'/'
      lidar_rmg_timestamps_path = lidar_rmg_path+'combined_lidar.timestamps'
      lidar_rmg_timestamps = np.loadtxt(lidar_rmg_timestamps_path, delimiter=' ', usecols=[0], dtype=np.float64)
      #print(lidar_rmg_timestamps.shape)

      # get first and last radar timestamp #
      radar_init_t = UnixTimeToSec(radar_timestamps[0])
      radar_end_t = UnixTimeToSec(radar_timestamps[np.size(radar_timestamps)-1])
      print("radar_init_t:")
      print(radar_init_t)
      print("radar_end_t:")
      print(radar_end_t)

      radar_idx = 0

      if MODE == 'test':
        frame_skip = 1
      elif MODE == 'train':
        frame_skip = 5

      northing = np.expand_dims(gps_data.iloc[:,5],axis=1)
      easting = np.expand_dims(gps_data.iloc[:,6],axis=1)
      position = np.concatenate((easting,northing),axis=1)
      print(position.shape)

      position_part1 = np.copy(position)
      position_part1 = np.delete(position_part1, np.where(position_part1[:,0]<620080), axis=0) #620104
      print(position_part1.shape)

      position_part2 = np.copy(position)
      position_part2 = np.delete(position_part2, np.where(position_part2[:,1]>5735325), axis=0)
      position_part2 = np.delete(position_part2, np.where(position_part2[:,0]>620080), axis=0)
      print(position_part2.shape)

      position_part3 = np.copy(position)
      position_part3 = np.delete(position_part3, np.where(position_part3[:,1]<5735325), axis=0)
      position_part3 = np.delete(position_part3, np.where(position_part3[:,0]>620080), axis=0)
      print(position_part3.shape)

      f, a = plt.subplots(1, 2, figsize=(10, 5))
      a[0].scatter(position[:,0], position[:,1], s=0.1)
      a[1].scatter(position_part1[:,0], position_part1[:,1], color='red', s=0.1)
      a[1].scatter(position_part2[:,0], position_part2[:,1], color='green', s=0.1)
      a[1].scatter(position_part3[:,0], position_part3[:,1], color='blue', s=0.1)

      print('len(radar_timestamps):', len(radar_timestamps))
      for radar_idx, radar_timestamp in enumerate(radar_timestamps, start=0):
        if radar_idx < history_scan_num or radar_idx%frame_skip!=0:
          print('skip', radar_idx, 'radar_idx < history_scan_num or radar_idx%frame_skip!=0')
          continue
        #print('------------------')
        #print(radar_idx)
        curr_radar_t = UnixTimeToSec(radar_timestamp)
        #print(curr_radar_t)
        lidar_res_idx, lidar_res_timestamp = get_sync(curr_radar_t, lidar_res_timestamps)
        lidar_rmg_idx, lidar_rmg_timestamp = get_sync(curr_radar_t, lidar_rmg_timestamps)
        if lidar_res_timestamp != lidar_rmg_timestamp:
          ## !!! TODO !!! This occur in several frame. It's due to the sync difference between ros msg flt and python sync... This might cause problem...
          print('skip', radar_idx, 'lidar_res_timestamp != lidar_rmg_timestamp', lidar_res_timestamp, lidar_rmg_timestamp)
          continue
        #assert lidar_res_timestamp == lidar_rmg_timestamp

        ### Load radar data ###
        radar_filename = radar_folder + str(radar_timestamps[radar_idx]) + '.png'
        radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
        radar_cart = polar_to_cart(radar_img).squeeze()

        radar_polar_list = load_radar_polar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, history_scan_num)

        if ODOM_COMP == True:
          radar_cart_list = load_radar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num)
        else:
          radar_cart_list = load_radar_data_wo_odom_comp(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num)

        ### Load sync data ###
        motion_seg_list, disp_0_list, disp_1_list, rmg_data_list = load_sync_data(curr_radar_t, lidar_res_timestamps, lidar_rmg_timestamps, lidar_res_path, lidar_rmg_path)

        ### Combined 5 lidar scans ###
        result_cart_radar_seg, combined_cart_seg, result_polar_radar_seg, combined_polar_seg = gen_gt_mask(motion_seg_list, radar_cart)
        result_cart_radar_disp_0 = gen_gt_vel(disp_0_list)
        result_cart_radar_disp_1 = gen_gt_vel(disp_1_list)
        #result_cart_radar_disp_0 = disp_0_list[0]
        #result_cart_radar_disp_1 = disp_1_list[0]
        result_cart_radar_rmg, combined_cart_rmg, result_polar_radar_seg, combined_polar_seg = gen_gt_mask(rmg_data_list, radar_cart)

        save_data_dict = dict()
        for i in range(len(radar_cart_list)):
          save_data_dict['raw_radar_' + str(i)] = radar_cart_list[i]

        save_data_dict['gt_radar'] = result_cart_radar_rmg
        save_data_dict['gt_moving'] = result_cart_radar_seg
        save_data_dict['gt_disp0'] = result_cart_radar_disp_0
        save_data_dict['gt_disp1'] = result_cart_radar_disp_1

        save_file_name = os.path.join(save_directory, str(radar_idx) + '.npy')
#        np.save(save_file_name, arr=save_data_dict)

#        ### Plot lidar motion segmetation result ###
#        merge_img = np.zeros((256,256,3))
#        lidar_msk = combined_cart_rmg == 1
#        merge_img[lidar_msk,:] = [1,1,1] # white
#        moving_mask = combined_cart_seg == 1
#        merge_img[moving_mask,:] = [1,0,0] # red

#        radar_merge_lidar_img = np.zeros((256,256,3))
#        radar_merge_lidar_img = np.stack((radar_cart,radar_cart,radar_cart), axis=2)
#        radar_merge_lidar_img = radar_merge_lidar_img*2.
#        radar_merge_lidar_img[lidar_msk,:] = [0,0,1] # blue
#        radar_merge_lidar_img[moving_mask,:] = [1,0,0] # red

#        ax[0].clear()
#        ax[0].imshow(radar_merge_lidar_img)
#        ax[0].axis('off')

#        viz_radar_img = np.zeros((256,256,3))
#        viz_radar_img = np.stack((radar_cart,radar_cart,radar_cart), axis=2)
#        viz_gt_rmg = np.zeros((256,256,3))
#        viz_gt_rmg[:,:,2] = result_cart_radar_rmg * np.logical_not(result_cart_radar_seg)
#        viz_gt_seg = np.zeros((256,256,3))
#        viz_gt_seg[:,:,0] = result_cart_radar_seg

#        ax[1].clear()
#        ax[1].imshow((viz_radar_img*2.+viz_gt_rmg+viz_gt_seg)/2)
#        ax[1].axis('off')

#        # Plot quiver.
#        field_gt = np.zeros((256,256,2))
#        field_gt[:,:,0] = result_cart_radar_disp_0
#        field_gt[:,:,1] = result_cart_radar_disp_1
#        idx_x = np.arange(field_gt.shape[0])
#        idx_y = np.arange(field_gt.shape[1])
#        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
#        # For cells with very small movements, we threshold them to be static
#        field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
#        #thd_mask = field_gt_norm <= 0.1
#        #field_gt[thd_mask, :] = 0
#        # Get the displacement field
#        mask = result_cart_radar_rmg.astype(np.bool) * result_cart_radar_seg.astype(np.bool)
#        X = idx_x[mask]
#        Y = idx_y[mask]
#        voxel_size = [0.25, 0.25]
#        U = field_gt[:, :, 0][mask] / 0.2#voxel_size[0]  # the distance between pixels is w.r.t. grid size (e.g., 0.2m)
#        V = -field_gt[:, :, 1][mask] / 0.2#voxel_size[1]
#        qk = ax[1].quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=1, width=0.001, headwidth=20, color='r')

#        ax_single[0].clear()
#        ax_single[0].imshow( save_data_dict['raw_radar_0'] )
#        ax_single[0].axis('off')
#        ax_single[1].clear()
#        ax_single[1].imshow( save_data_dict['raw_radar_1'] )
#        ax_single[1].axis('off')
#        ax_single[2].clear()
#        ax_single[2].imshow( save_data_dict['raw_radar_2'] )
#        ax_single[2].axis('off')
#        ax_single[3].clear()
#        ax_single[3].imshow( save_data_dict['raw_radar_3'] )
#        ax_single[3].axis('off')
#        ax_single[4].clear()
#        ax_single[4].imshow( save_data_dict['raw_radar_4'] )
#        ax_single[4].axis('off')

#        plt.pause(0.01)
#        time.sleep(0.05)
      #plt.show()






if __name__ == '__main__':
  main()
