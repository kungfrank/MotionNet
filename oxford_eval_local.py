# python3 no ros

# ins/ro frame:
#   x -> top
#   y -> right
# disp frame:
#   disp0, x -> right
#   disp1, y -> top
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from oxford_model import RaMNet
#import rospy

import os
pid = os.getpid()
print('pid:', pid)

MASKING=False
plot_motion_seg=True
SPATIAL_VAL = False

which_model = 'RaMNet'
trained_model_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-04-09_16-28-24/epoch_10.pth' # 2021-03-23_23-22-37/epoch_10.pth'

sec1_model_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-04-07_17-21-17/epoch_20.pth'
sec2_model_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-04-07_19-50-50/epoch_20.pth'
sec3_model_path = '/home/joinet/MotionNet/trained_model/train_multi_seq/2021-04-07_22-33-01/epoch_20.pth'

log_name = '2019-01-10-12-32-52-radar-oxford-10k' # '2019-01-10-11-46-21-radar-oxford-10k' '2019-01-10-12-32-52-radar-oxford-10k'

frame_skip = 1

history_scan_num = 2
num_past_frames = 2
out_seq_len = 1
height_feat_size=1
use_temporal_info = True

cart_resolution = 0.25
cart_pixel_width = 256
interpolate_crossover = True

fig, ax = plt.subplots(1, 2, figsize=(5, 5))
plt.tight_layout()
#ax[1].set_xlim(-600, 600)
#ax[1].set_ylim(-600, 600)

global gt_odom_tf, res_odom_tf
gt_odom_tf = np.identity(3)#.astype(np.float64)
res_odom_tf = np.identity(3)#.astype(np.float64)

global err_delta_x_sum, err_delta_y_sum, err_delta_theta_sum
err_delta_x_sum = 0
err_delta_y_sum = 0
err_delta_theta_sum = 0

#gt_odom_file = open("./odometry_file/gt_odom_.txt","w+")
#res_odom_file = open("./odometry_file/2021-04-06_17-28-47_res_test.txt","w+")

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
      thd_mask = vel_map < 0.5 # 0.5 pixel / s
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
  return result_cart_radar, combined_cart_lidar # with mask, without mask

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
  tf_mat_list = list()
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
    radar_cart_list.append(radar_cart)
    tf_mat_list.append(tf_mat)
  return radar_cart_list, tf_mat_list

def gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg):
  N=256
  static_disp_map_list = list()

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
    static_disp_map = np.zeros((N,N,2))
    for i in range(N): # row
      for j in range(N): # col
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
    static_disp_map_list.append(static_disp_map*result_cart_radar_rmg.astype(np.bool).reshape((N,N,1)))
  return static_disp_map_list


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

def calc_odom_by_disp_map(disp0, disp1, radar_mask, moving_mask, cart_resolution):
  N=256
  center=N/2-0.5

  radar_mask[radar_mask>0] = 1
  radar_mask[radar_mask<0] = 0
  valid_mask = radar_mask.astype(np.bool) # (N x N)

  N_vec = np.arange(N).reshape(1,-1)
  c_mat = np.full((N, N), center)

  # x, y w.r.t. ego-motion frame
  point_x = c_mat - N_vec.T
  point_y = -(c_mat - N_vec.T).T
  point_x = np.expand_dims(point_x, axis=2)
  point_y = np.expand_dims(point_y, axis=2)
  point = np.concatenate((point_x, point_y), axis=2) # (N x N x 2)

  disp_map = np.zeros((256,256,2)) # (N x N x 2)
  disp_map[:,:,0] = disp1
  disp_map[:,:,1] = disp0

  point_trans = point + disp_map # (N x N x 2)

  point = point.reshape(-1, 2) # (N^2 x 2)
  point_trans = point_trans.reshape(-1, 2) # (N^2 x 2)
  valid_mask = valid_mask.reshape(-1, 1) # (N^2 x 1)

  # mask by valid mask
  point = np.delete(point, np.where(valid_mask==0), axis=0)
  point_trans = np.delete(point_trans, np.where(valid_mask==0), axis=0)

  pc = np.concatenate((point, np.zeros((point.shape[0], 1))), axis=1)
  pc_ = np.concatenate((point_trans, np.zeros((point_trans.shape[0], 1))), axis=1)
  M, mask = cv2.findHomography(point, point_trans, cv2.RANSAC, 3.0) # 1~10 => strict~loose # result tf is w.r.t. disp frame

  M[0,2] = M[0,2]*cart_resolution
  M[1,2] = M[1,2]*cart_resolution
  return M

def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat


def viz_combined(img, denoised_img, motion_seg):
  viz_img = np.zeros((256,256,3))
  viz_img = np.stack((img,img,img), axis=2)
  viz_denoised_img = np.zeros((256,256,3))
  if plot_motion_seg:
    viz_denoised_img[:,:,2] = denoised_img * np.logical_not(motion_seg)
    viz_seg = np.zeros((256,256,3))
    viz_seg[:,:,0] = motion_seg
    return (viz_img*2+viz_denoised_img+viz_seg)/2.
  else:
    viz_denoised_img[:,:,2] = denoised_img
    return (viz_img*2+viz_denoised_img)/2.

def main():

  global gt_odom_tf, res_odom_tf
  global err_delta_x_sum, err_delta_y_sum, err_delta_theta_sum

  ##### Load Model #####
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # Load pre-trained network weights
  loaded_models = list()
  if which_model == "RaMNet":
    if SPATIAL_VAL:
      model1 = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
      model1 = nn.DataParallel(model1)
      checkpoint1 = torch.load(sec1_model_path)
      model1.load_state_dict(checkpoint1['model_state_dict'], False)
      model1 = model1.to(device)

      model2 = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
      model2 = nn.DataParallel(model2)
      checkpoint2 = torch.load(sec2_model_path)
      model2.load_state_dict(checkpoint2['model_state_dict'], False)
      model2 = model2.to(device)

      model3 = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
      model3 = nn.DataParallel(model3)
      checkpoint3 = torch.load(sec3_model_path)
      model3.load_state_dict(checkpoint3['model_state_dict'], False)
      model3 = model3.to(device)

      loaded_models = [model1, model2, model3]
    else:
      model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
      model = nn.DataParallel(model)
      checkpoint = torch.load(trained_model_path)
      model.load_state_dict(checkpoint['model_state_dict'], False)
      model = model.to(device)
      loaded_models = [model]
  else:
      print('model error')
  print("Loaded pretrained model {}".format(which_model))

  ##### Read Data #####
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

    lidar_rmg_path = '/mnt/Disk1/oxford_rmg_lidar_combined_ts_from_radar/'+log_name+'/'
    lidar_rmg_timestamps_path = lidar_rmg_path+'combined_lidar.timestamps'
    lidar_rmg_timestamps = np.loadtxt(lidar_rmg_timestamps_path, delimiter=' ', usecols=[0], dtype=np.float64)

    # get first and last radar timestamp #
    radar_init_t = UnixTimeToSec(radar_timestamps[0])
    radar_end_t = UnixTimeToSec(radar_timestamps[np.size(radar_timestamps)-1])
    print("radar_init_t:")
    print(radar_init_t)
    print("radar_end_t:")
    print(radar_end_t)

    radar_idx = 0
    count=0
    print('len(radar_timestamps):', len(radar_timestamps))
    for radar_idx, radar_timestamp in enumerate(radar_timestamps, start=0):
      print('--------')
      print(radar_idx)
      if radar_idx < history_scan_num or radar_idx%frame_skip!=0 or radar_idx<0:# or radar_idx>4000: # 1450::turning
        #print('skip', radar_idx, 'radar_idx < history_scan_num or radar_idx%frame_skip!=0')
        continue
      curr_radar_t = UnixTimeToSec(radar_timestamp)
      lidar_res_idx, lidar_res_timestamp = get_sync(curr_radar_t, lidar_res_timestamps)
      lidar_rmg_idx, lidar_rmg_timestamp = get_sync(curr_radar_t, lidar_rmg_timestamps)

      gps_idx, gps_timestamp = get_sync(radar_timestamp, gps_timestamps)
#      if lidar_res_timestamp != lidar_rmg_timestamp:
#        ## !!! TODO !!! This occur in several frame. It's due to the sync difference between ros msg flt and python sync... This might cause problem...
#        print('skip', radar_idx, 'lidar_res_timestamp != lidar_rmg_timestamp', lidar_res_timestamp, lidar_rmg_timestamp)
#        continue
      #assert lidar_res_timestamp == lidar_rmg_timestamp

      count+=1

      ### Load radar & odom data ###
      radar_filename = radar_folder + str(radar_timestamps[radar_idx]) + '.png'
      radar_img = cv2.imread(radar_filename, cv2.IMREAD_GRAYSCALE)
      radar_cart = polar_to_cart(radar_img).squeeze()
      radar_cart_list, tf_mat_list = load_radar_data(radar_folder, radar_timestamps, radar_idx, ro_timestamps, ro_data, history_scan_num)
      ### Load sync data ###
      motion_seg_list, disp_0_list, disp_1_list, rmg_data_list = load_sync_data(curr_radar_t, lidar_res_timestamps, lidar_rmg_timestamps, lidar_res_path, lidar_rmg_path)
      ### Combined 5 lidar scans ###
      result_cart_radar_seg, combined_cart_seg = gen_gt_mask(motion_seg_list, radar_cart)
      result_cart_radar_disp_0 = gen_gt_vel(disp_0_list)
      result_cart_radar_disp_1 = gen_gt_vel(disp_1_list)
      result_cart_radar_rmg, combined_cart_rmg = gen_gt_mask(rmg_data_list, radar_cart)
      ### Generate disp by motion ###
#      result_disp_map_seq_list = gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg)
      ### Generate gt dict ###
      data_dict = dict()
      for i in range(len(radar_cart_list)):
        data_dict['raw_radar_' + str(i)] = radar_cart_list[i]
      for i in range(len(tf_mat_list)):
        data_dict['tf_mat_' + str(i)] = tf_mat_list[i]
#      for i in range(len(result_disp_map_seq_list)):
#        data_dict['gt_disp_' + str(i)] = result_disp_map_seq_list[i]
      data_dict['gt_radar'] = result_cart_radar_rmg
      data_dict['gt_moving'] = result_cart_radar_seg
      data_dict['gt_lidar'] = combined_cart_rmg

      ### Testing...
      #print('Testing...')
      raw_radars = list()
      for i in range(num_past_frames):
        raw_radars.append(np.expand_dims(data_dict['raw_radar_' + str(i)], axis=2))
      raw_radars = np.stack(raw_radars, 0).astype(np.float32)
      raw_radars_list = list()
      raw_radars_list.append(raw_radars)
      raw_radars = np.stack(raw_radars_list, 0)
      raw_radars = torch.tensor(raw_radars) # torch.Size([1, 2, 256, 256, 1])

      raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
      raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
      raw_radars_curr = raw_radars_curr.to(device)

      tf_mat_list = list()
      for i in range(num_past_frames):
        tf_mat_list.append(data_dict['tf_mat_' + str(i)])
      #print(tf_mat_list[1])

      gt_delta_tf = tf_mat_list[1] # w.r.t. disp frame
      delta_x = gt_delta_tf[0,2]
      delta_y = gt_delta_tf[1,2]
      delta_yaw = np.arctan(gt_delta_tf[1,0] / gt_delta_tf[0,0])
      #print('gt: ', delta_x, delta_y, delta_yaw) # w.r.t. disp frame


      if SPATIAL_VAL:
        northing = gps_data.iloc[gps_idx,5]
        easting = gps_data.iloc[gps_idx,6]

        model1 = loaded_models[0]
        model1.eval()
        model2 = loaded_models[1]
        model2.eval()
        model3 = loaded_models[2]
        model3.eval()

        if easting > 620080:
          section = 1
          with torch.no_grad():
            if use_temporal_info:
              disp_pred, cat_pred, motion_pred = model1(raw_radars)
            else:
              disp_pred, cat_pred, motion_pred = model1(raw_radars_curr)
        elif northing < 5735325:
          section = 2
          with torch.no_grad():
            if use_temporal_info:
              disp_pred, cat_pred, motion_pred = model2(raw_radars)
            else:
              disp_pred, cat_pred, motion_pred = model2(raw_radars_curr)
        else:
          section = 3
          with torch.no_grad():
            if use_temporal_info:
              disp_pred, cat_pred, motion_pred = model3(raw_radars)
            else:
              disp_pred, cat_pred, motion_pred = model3(raw_radars_curr)
        print('Section:', section)

      else:
        model = loaded_models[0]
        model.eval()
        with torch.no_grad():
          if use_temporal_info:
            disp_pred, cat_pred, motion_pred = model(raw_radars)
          else:
            disp_pred, cat_pred, motion_pred = model(raw_radars_curr)

      # convert all output to numpy
      cat_pred_numpy = cat_pred.cpu().numpy()
      motion_pred_numpy = motion_pred.cpu().numpy()
      disp_pred_numpy = disp_pred.cpu().numpy()
      raw_radars = raw_radars.detach().numpy()

#      f, a = plt.subplots(1, 1, figsize=(5, 5))
#      a.imshow(motion_pred_numpy[0,0,:,:])
#      plt.show()

      res_delta_tf = calc_odom_by_disp_map(disp_pred_numpy[0,0], disp_pred_numpy[0,1], cat_pred_numpy[0,0,:,:], motion_pred_numpy[0,0,:,:], cart_resolution) # w.r.t. disp frame
      #res_delta_tf = calc_odom_by_disp_map(result_disp_map_seq_list[0][:,:,0], result_disp_map_seq_list[0][:,:,1], result_cart_radar_rmg, result_cart_radar_seg, cart_resolution)

#      delta_x_res = res_delta_tf[0,2]
#      delta_y_res = res_delta_tf[1,2]
#      delta_yaw_res = np.arctan(res_delta_tf[1,0] / res_delta_tf[0,0])
#      err_delta_x = abs(delta_x_res-delta_x)
#      err_delta_y = abs(delta_y_res-delta_y)
#      err_delta_yaw = abs(delta_yaw_res-delta_yaw)
#      err_delta_x_sum += err_delta_x
#      err_delta_y_sum += err_delta_y
#      err_delta_theta_sum += err_delta_yaw
#      gt_delta_tf_new = GetMatFromXYYaw(delta_x,delta_y,delta_yaw)
#      res_delta_tf_new = GetMatFromXYYaw(delta_x_res,delta_y_res,delta_yaw_res)

      #print(raw_radars.shape)
      #print(raw_radars.squeeze()[0].shape)
      ax[0].clear
      ax[0].imshow(raw_radars.squeeze()[0])
      ax[1].imshow(viz_combined(raw_radars.squeeze()[0], result_cart_radar_rmg, result_cart_radar_seg))
      plt.pause(0.1)

      #np.set_printoptions(precision=4, suppress=True)
      #print(gt_delta_tf)
      #print(res_delta_tf)

#      gt_odom_tf = gt_odom_tf.dot(gt_delta_tf_new)
#      res_odom_tf = res_odom_tf.dot(res_delta_tf_new)

#      ax[1].clear
#      rot = np.identity(3)
#      rot[:2,:2] = gt_odom_tf[:2,:2]
#      r = R.from_matrix(rot)
#      yaw = r.as_euler('xyz', degrees=False)[2]
#      X = gt_odom_tf[0,2]
#      Y = gt_odom_tf[1,2]
#      U = 1 * np.cos(yaw)
#      V = 1 * np.sin(yaw)
#      qk1 = ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r', alpha=1) # width=0.001, headwidth=20

#      rot = np.identity(3)
#      rot[:2,:2] = res_odom_tf[:2,:2]
#      r = R.from_matrix(rot)
#      yaw = r.as_euler('xyz', degrees=False)[2]
#      X = res_odom_tf[0,2]
#      Y = res_odom_tf[1,2]
#      U = 1 * np.cos(yaw)
#      V = 1 * np.sin(yaw)
#      qk2 = ax[1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='b', alpha=1) # width=0.001, headwidth=20

#      plt.pause(0.01)
      #plt.show()

#      gt_odom_file.write("%f %f %f %f %f %f %f %f %f %f %f %f\n"\
#                      % (gt_odom_tf[0,0], gt_odom_tf[0,1], 0, gt_odom_tf[0,2],\
#                         gt_odom_tf[1,0], gt_odom_tf[1,1], 0, gt_odom_tf[1,2],\
#                         0, 0, 1, 0))
#      res_odom_file.write("%f %f %f %f %f %f %f %f %f %f %f %f\n"\
#                     % (res_odom_tf[0,0], res_odom_tf[0,1], 0, res_odom_tf[0,2],\
#                        res_odom_tf[1,0], res_odom_tf[1,1], 0, res_odom_tf[1,2],\
#                        0, 0, 1, 0))


  plt.show()



if __name__ == '__main__':
  main()
