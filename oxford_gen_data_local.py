# python3 no ros

# ins/ro frame:
#   x -> top
#   y -> right
# disp frame:
#   disp0, x -> right
#   disp1, y -> top

import argparse
import pandas as pd
import time
import os
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import time
import math

from scipy.spatial.transform import Rotation as R
import time
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import open3d as o3d
import copy

import os
pid = os.getpid()
print('pid:', pid)

log_name = '2019-01-10-11-46-21-radar-oxford-10k' # '2019-01-10-11-46-21-radar-oxford-10k' '2019-01-10-12-32-52-radar-oxford-10k'

MODE = 'train' # 'test' 'train'
MASKING = False
SPATIAL_CROSS_VALIDATION = False

save_directory = '/mnt/Disk1/training_data_local_global_vel_i2o1_no_mask_polar/'+log_name

history_scan_num = 2 # 2
out_seq_len = 1 # history_scan_num - 1

cart_resolution = 0.25
cart_pixel_width = 256
interpolate_crossover = True

fig, ax = plt.subplots(1, 2, figsize=(20, 8))
#fig_single, ax_single = plt.subplots(1, 5, figsize=(20, 8))
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
    #fft_data[:,740] = 1.
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
      thd_mask = vel_map < 0.05 # 0.5 # !!!!!!!!!!!!!!!!! 0.5 is used in initial test, but this is too big to segment moving pedestrian
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

#def load_sync_labeled_data(radar_idx, lidar_res_timestamps, lidar_rmg_timestamps, lidar_res_path, lidar_rmg_path):

def img_cart_to_polar_extend(img):
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

def img_polar_to_cart_extend(img):
  img = np.concatenate((img[int(img.shape[0]/4):,:], img[:int(img.shape[0]/4),:]), axis=0)
  value = img.shape[0]/2.0
  cart_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
  cart_image = cv2.resize(cart_image, (256, 256), interpolation=cv2.INTER_LINEAR)
  return cart_image

def img_cart_to_polar(img):
  img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
  img = np.rot90(img,-1) # rot clockwise 90 deg
  value = img.shape[0]/2.0
  polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
  return polar_image

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
    polar_lidar = img_cart_to_polar_extend(data_list[i])
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

  radar_polar = img_cart_to_polar_extend(radar_cart)

  if MASKING == True:
    # masking #
    result_polar_radar = mask_radar_by_lidar(radar_polar, combined_polar_lidar)
  else:
    # no masking #
    result_polar_radar = combined_polar_lidar

  result_cart_radar = img_polar_to_cart(result_polar_radar)

  combined_cart_lidar = img_polar_to_cart(combined_polar_lidar)

  combined_polar_lidar_small = img_cart_to_polar(combined_cart_lidar)
  result_polar_radar_small = combined_polar_lidar_small

  combined_cart_lidar[combined_cart_lidar<0.5] = 0
  combined_cart_lidar[combined_cart_lidar>0.5] = 1

  combined_polar_lidar_small[combined_polar_lidar_small<0.5] = 0
  combined_polar_lidar_small[combined_polar_lidar_small>0.5] = 1
  result_polar_radar_small[result_polar_radar_small<0.5] = 0
  result_polar_radar_small[result_polar_radar_small>0.5] = 1

  return result_cart_radar, combined_cart_lidar, result_polar_radar_small, combined_polar_lidar_small # with mask, without mask
  ### polar resolution = 32*sqrt(2)/260

def gen_gt_vel(disp_list):
  combined_polar_lidar = combined_lidar_scans(disp_list)
  result_cart_radar = img_polar_to_cart(combined_polar_lidar)

  result_polar_radar_small = img_cart_to_polar(result_cart_radar)

  return result_cart_radar, result_polar_radar_small

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
  radar_polar_list = list()
  tf_mat_list = list()
  #print('radar_idx',radar_idx)
  for i in range(history_scan_num):
    #print('i',i)
    idx = radar_idx - i
    radar_filename = radar_folder + str(radar_timestamps[idx]) + '.png'
    radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
    radar_polar = radar_img[:, 11:]
    radar_polar = cv2.resize(radar_polar, (1323, 260)) # 935 = 3768 * 0.0432 / (32/260)

    radar_cart = polar_to_cart(radar_img).squeeze()
    tf_mat = np.identity(3).astype(np.float32)
    for j in range(i):
      j = i-1-j
      ro_idx = radar_idx-1-j
      #print('ro_idx',ro_idx)
      ro_tf = GetMatFromXYYaw(ro_data.iloc[ro_idx,2],ro_data.iloc[ro_idx,3],ro_data.iloc[ro_idx,7])
      tf_mat = tf_mat.dot(ro_tf)
    radar_cart_list.append(radar_cart)
    radar_polar_list.append(radar_polar)
    tf_mat_list.append(tf_mat) # from old to new
  return radar_cart_list, radar_polar_list, tf_mat_list

def viz_err_fig(raw_radar, gt, res):
  viz_err = np.zeros((raw_radar.shape[0], raw_radar.shape[1], 3))
  viz_err[:,:,0] = gt * np.logical_not(res)
  viz_err[:,:,1] = gt * res
  viz_err[:,:,2] = res * np.logical_not(gt)
  viz_radar = np.stack((raw_radar,raw_radar,raw_radar), axis=2)
  return (viz_err + viz_radar*3)/2.

def gen_static_disp_map_old(tf_mat_list, result_cart_radar_rmg, out_seq_len):
  print('gen_static_disp_map_old')
  N=256
  static_disp_map_list = list()
  assert len(tf_mat_list)>out_seq_len
  for idx in range(out_seq_len):
    tf_mat = tf_mat_list[idx+1] # ego-motion w.r.t ins/ro frame
    delta_x = tf_mat[0,2]/cart_resolution
    delta_y = tf_mat[1,2]/cart_resolution
    delta_yaw = np.arctan(tf_mat[1,0] / tf_mat[0,0])

    delta_x = 2
    delta_y = 0
    delta_yaw = 0.05

    delta_tf = GetMatFromXYYaw(delta_x, delta_y, delta_yaw)
    print('delta:', delta_x, delta_y, delta_yaw)
    static_disp_map = np.zeros((N,N,2))
    for i in range(N): # row
      for j in range(N): # col
        center=N/2-0.5
        local_tf_curr_t = GetMatFromXYYaw(center-i, j-center, 0)
        local_tf_old_t = np.dot(delta_tf, local_tf_curr_t)
        x_mo = local_tf_old_t[0,2] - local_tf_curr_t[0,2]
        y_mo = local_tf_old_t[1,2] - local_tf_curr_t[1,2]
        static_disp_map[i,j,0] = y_mo # x-axis of disp frame
        static_disp_map[i,j,1] = x_mo # y-axis of disp frame
    static_disp_map_list.append(static_disp_map*result_cart_radar_rmg.astype(np.bool).reshape((N,N,1)))
  # debug
  fig_, ax_ = plt.subplots(1, 1, figsize=(8, 8))
  field_gt = static_disp_map_list[0]
  idx_x = np.arange(field_gt.shape[0])
  idx_y = np.arange(field_gt.shape[1])
  idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
  #mask = np.ones((256,256)).astype(np.bool)
  #mask[120:136,120:136] = 1
  mask = result_cart_radar_rmg.astype(np.bool)
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask] / 0.2
  V = -field_gt[:, :, 1][mask] / 0.2
  qk = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=3, width=0.001, headwidth=20, color='r')
  plt.show()

  return static_disp_map_list

def gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg):
  N=256
  static_cart_disp_map_list = list()
  static_polar_disp_map_list = list()

  field_gt = np.zeros((N,N,2))
  field_gt[:,:,0] = result_cart_radar_disp_0
  field_gt[:,:,1] = result_cart_radar_disp_1
  mask = result_cart_radar_rmg.astype(np.bool) * result_cart_radar_seg.astype(np.bool)
  #mask = np.zeros((256,256))
  field_gt = field_gt*mask.reshape((N,N,1))

  assert len(tf_mat_list)>out_seq_len
  for idx in range(out_seq_len):
    tf_mat = tf_mat_list[idx+1] # ego-motion w.r.t ins/ro frame

    # transform ego-motion to pixel unit and w.r.t disp frame
    #print('delta (m) :', tf_mat[0,2], tf_mat[1,2], np.arctan(tf_mat[1,0] / tf_mat[0,0]))
    delta_x = tf_mat[1,2]/cart_resolution
    delta_y = tf_mat[0,2]/cart_resolution
    delta_yaw = -np.arctan(tf_mat[1,0] / tf_mat[0,0])

    delta_tf = GetMatFromXYYaw(delta_x, delta_y, delta_yaw) # w.r.t disp frame
    #print('delta:', delta_x, delta_y, delta_yaw)

    valid_mask = result_cart_radar_rmg.astype(np.bool) # only compute disp for valid pixel
    static_disp_map = np.zeros((N,N,2))
    for i in range(N): # row
      for j in range(N): # col
        if valid_mask[i,j] == 1:
          center=N/2-0.5
          local_tf_curr_t = GetMatFromXYYaw(j-center, center-i, 0)
          disp_wrt_curr_t = field_gt[i,j,:]*4 ### ????????????????????????????????????????????????? Make sure this is correct...
          #print(disp_wrt_curr_t)
          disp_tf_wrt_curr_t = GetMatFromXYYaw(disp_wrt_curr_t[0], disp_wrt_curr_t[1], 0)
          local_tf_old_t = delta_tf.dot(inv(disp_tf_wrt_curr_t.dot(inv(local_tf_curr_t))))

          #local_tf_old_t = np.dot(delta_tf, local_tf_curr_t)
          x_mo = local_tf_old_t[0,2] - local_tf_curr_t[0,2]
          y_mo = local_tf_old_t[1,2] - local_tf_curr_t[1,2]
          static_disp_map[i,j,0] = x_mo # x-axis of disp frame
          static_disp_map[i,j,1] = y_mo # y-axis of disp frame
    cart_disp_map = static_disp_map*result_cart_radar_rmg.astype(np.bool).reshape((N,N,1))

    polar_disp0_map_small = img_cart_to_polar(cart_disp_map[:,:,0])
    polar_disp1_map_small = img_cart_to_polar(cart_disp_map[:,:,1])
    polar_disp_map_small = np.concatenate((np.expand_dims(polar_disp0_map_small, axis=2), np.expand_dims(polar_disp1_map_small, axis=2)), axis=2)

    static_cart_disp_map_list.append(cart_disp_map)
    static_polar_disp_map_list.append(polar_disp_map_small)
  return static_cart_disp_map_list, static_polar_disp_map_list


def gen_corr_line_set(src, dst, corres, color):
    viz_points = np.concatenate((src, dst), axis=1)
    viz_lines = list()
    for corr in corres:
      associate_shift = corr[1]-corr[0]
      viz_lines.append([corr[0],corr[0]+src.shape[1]])
    colors = [color for i in range(len(viz_lines))]
    line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(viz_points.T),
            lines=o3d.utility.Vector2iVector(viz_lines),
        )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def calc_odom_by_disp_map(disp_map, radar_mask, moving_mask, lidar_mask):
  pointcloud = list()
  pointcloud_ = list()
  N=256
  center=N/2-0.5
  for i in range(N): # row
    for j in range(N): # col
      if radar_mask[i,j]>0.5:
        point = np.array([center-i, j-center, 0]) # x, y in ego-motion frame
        delta = np.array([disp_map[i,j,1], disp_map[i,j,0], 0])
        point_ = point + delta
        pointcloud.append(point)
        pointcloud_.append(point_)
  pc = np.array(pointcloud)
  pc_ = np.array(pointcloud_)
  #print(pc.shape)
  #print(pc_.shape)
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc)
  pcd.paint_uniform_color([1, 0, 0])
  pcd_ = o3d.geometry.PointCloud()
  pcd_.points = o3d.utility.Vector3dVector(pc_)
  pcd_.paint_uniform_color([0, 1, 0])
  arr = np.expand_dims(np.arange(pc.shape[0]),axis=0)
  np_corres = np.concatenate((arr, arr), axis=0).T
  corres = o3d.utility.Vector2iVector(np_corres)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  M, mask = cv2.findHomography(pc, pc_, cv2.RANSAC, 5.0)
  print(M[1,2], M[0,2], -np.arctan(M[1,0]/M[0,0]))
  np_corres_new = np_corres[mask.squeeze().astype(np.bool),:]
  corres_new = o3d.utility.Vector2iVector(np_corres_new)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres_new, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])

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
      gps_timestamps = np.array(gps_data.iloc[:,0].copy())

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
      print(radar_timestamps[0])
      print(radar_timestamps[np.size(radar_timestamps)-1])
      print("radar_init_t:")
      print(radar_init_t)
      print("radar_end_t:")
      print(radar_end_t)

      radar_idx = 0

      if SPATIAL_CROSS_VALIDATION:
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
#        f, a = plt.subplots(1, 2, figsize=(10, 5))
#        a[0].scatter(position[:,0], position[:,1], s=0.1)
#        a[1].scatter(position_part1[:,0], position_part1[:,1], color='red', s=0.001)
#        a[1].scatter(position_part2[:,0], position_part2[:,1], color='green', s=0.001)
#        a[1].scatter(position_part3[:,0], position_part3[:,1], color='blue', s=0.001)

      if MODE == 'test':
        frame_skip = 1
      elif MODE == 'train':
        frame_skip = 2
      elif MODE == 'view':
        frame_skip = 4

      print('len(radar_timestamps):', len(radar_timestamps))
      for radar_idx, radar_timestamp in enumerate(radar_timestamps, start=0):
        if radar_idx < history_scan_num or radar_idx%frame_skip!=0 or radar_idx<0:
          #print('skip', radar_idx, 'radar_idx < history_scan_num or radar_idx%frame_skip!=0')
          continue
        print('------------------')
        print(radar_idx)
        curr_radar_t = UnixTimeToSec(radar_timestamp)
        #print(radar_timestamp, curr_radar_t)
        lidar_res_idx, lidar_res_timestamp = get_sync(curr_radar_t, lidar_res_timestamps)
        lidar_rmg_idx, lidar_rmg_timestamp = get_sync(curr_radar_t, lidar_rmg_timestamps)

        gps_idx, gps_timestamp = get_sync(radar_timestamp, gps_timestamps)
        #print(gps_timestamp, UnixTimeToSec(gps_timestamp))

        if lidar_res_timestamp != lidar_rmg_timestamp:
          ## !!! TODO !!! This occur in several frame. It's due to the sync difference between ros msg flt and python sync... This might cause problem...
          print('skip', radar_idx, 'lidar_res_timestamp != lidar_rmg_timestamp', lidar_res_timestamp, lidar_rmg_timestamp)
          continue
        #assert lidar_res_timestamp == lidar_rmg_timestamp

        ### Load radar data ###
        radar_filename = radar_folder + str(radar_timestamps[radar_idx]) + '.png'
        radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
        radar_cart = polar_to_cart(radar_img).squeeze()

        radar_cart_list, radar_polar_list, tf_mat_list = load_radar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num)

        ### Find corresponding section ###
        if SPATIAL_CROSS_VALIDATION:
          section = 0
          northing = gps_data.iloc[gps_idx,5]
          easting = gps_data.iloc[gps_idx,6]
          if easting > 620080:
            section = 1
            #a[1].scatter(easting, northing, color='red', s=10)
          elif northing < 5735325:
            section = 2
            #a[1].scatter(easting, northing, color='green', s=10)
          else:
            section = 3
            #a[1].scatter(easting, northing, color='blue', s=10)
          print('Section:', section)

#        plt.show()
#        exit(0)

        ### Load sync data ###
        motion_seg_list, disp_0_list, disp_1_list, rmg_data_list = load_sync_data(curr_radar_t, lidar_res_timestamps, lidar_rmg_timestamps, lidar_res_path, lidar_rmg_path)

        ### Combined 5 lidar scans ###
        result_cart_radar_seg, combined_cart_seg, result_polar_radar_seg, combined_polar_seg = gen_gt_mask(motion_seg_list, radar_cart)
        result_cart_radar_disp_0, result_polar_radar_disp_0 = gen_gt_vel(disp_0_list)
        result_cart_radar_disp_1, result_polar_radar_disp_1 = gen_gt_vel(disp_1_list)
        result_cart_radar_rmg, combined_cart_rmg, result_polar_radar_rmg, combined_polar_rmg = gen_gt_mask(rmg_data_list, radar_cart)

        ### Generate disp by motion ###
        result_disp_map_seq_list, result_disp_polar_map_seq_list = gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg)


        save_data_dict = dict()
        for i in range(len(radar_polar_list)):
          save_data_dict['raw_radar_' + str(i)] = radar_polar_list[i] # radar_cart_list[i]

        for i in range(len(result_disp_map_seq_list)):
          save_data_dict['gt_disp_' + str(i)] = result_disp_polar_map_seq_list[i] # result_disp_map_seq_list[i]

#        save_data_dict['gt_disp_global'] = np.concatenate((np.expand_dims(result_cart_radar_disp_0,axis=2), np.expand_dims(result_cart_radar_disp_1,axis=2)), axis=2)
        save_data_dict['gt_disp_global'] = np.concatenate((np.expand_dims(result_polar_radar_disp_0,axis=2), np.expand_dims(result_polar_radar_disp_1,axis=2)), axis=2)

#        for i in range(len(tf_mat_list)): # store as tf w.r.t. disp frame
#          tf_mat = tf_mat_list[i]
#          delta_x = tf_mat[1,2]
#          delta_y = tf_mat[0,2]
#          delta_yaw = -np.arctan(tf_mat[1,0] / tf_mat[0,0])
#          save_data_dict['gt_tf_' + str(i)] = [delta_x, delta_y, delta_yaw]
          #print('tf_mat: ', delta_x, delta_y, delta_yaw)
        save_data_dict['gt_radar'] = result_polar_radar_rmg # result_cart_radar_rmg
        save_data_dict['gt_moving'] = result_polar_radar_seg # result_cart_radar_seg
        save_data_dict['gt_lidar'] = combined_polar_rmg # combined_cart_rmg

        if SPATIAL_CROSS_VALIDATION:
          save_file_name = os.path.join(save_directory+'/sec'+str(section), str(radar_idx) + '.npy')
        else:
          save_file_name = os.path.join(save_directory, str(radar_idx) + '.npy')
        #np.save(save_file_name, arr=save_data_dict)

        viz_img = viz_err_fig(radar_polar_list[0][:,:260].astype(np.float32)/255., combined_polar_rmg, result_polar_radar_seg)
        f, a = plt.subplots(1, 2, figsize=(20, 8))
        a[0].imshow(viz_img)
        a[1].imshow(radar_polar_list[0][:,260:260*2], cmap='gray')
        #a[0].imshow(save_data_dict['gt_disp_global'][:,:,0]*save_data_dict['gt_lidar'])
        #a[1].imshow(save_data_dict['gt_disp_global'][:,:,1]*save_data_dict['gt_lidar'])
        plt.tight_layout()
        a[0].axis('off')
        a[0].set_aspect('equal')
        a[1].axis('off')
        a[1].set_aspect('equal')
        plt.show()

        ### Plot lidar motion segmetation result ###
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

#        field_gt = result_disp_map_seq_list[0]

#        idx_x = np.arange(field_gt.shape[0])
#        idx_y = np.arange(field_gt.shape[1])
#        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')

#        mask = result_cart_radar_rmg.astype(np.bool) * result_cart_radar_seg.astype(np.bool) * combined_cart_rmg.astype(np.bool) #  * combined_cart_rmg.astype(np.bool) is better !!
#        X = idx_x[mask]
#        Y = idx_y[mask]
#        U = field_gt[:, :, 0][mask]
#        V = -field_gt[:, :, 1][mask]
#        qk = ax[1].quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.2, width=0.0015, headwidth=10, color='r')

#        mask = result_cart_radar_rmg.astype(np.bool) * np.logical_not(result_cart_radar_seg.astype(np.bool)) * combined_cart_rmg.astype(np.bool) #  * combined_cart_rmg.astype(np.bool) is better !!
#        X = idx_x[mask]
#        Y = idx_y[mask]
#        U = field_gt[:, :, 0][mask]
#        V = -field_gt[:, :, 1][mask]
#        qk = ax[1].quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.2, width=0.0015, headwidth=10, color='b')

        #img_save_dir = '/mnt/Disk1/viz_result/motion_img'
        #plt.savefig(os.path.join(img_save_dir, str(radar_idx) + '.png'))

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

        plt.pause(0.01)
        time.sleep(0.01)
      plt.show()




if __name__ == '__main__':
  main()
