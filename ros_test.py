
import argparse
import pandas as pd
import time
import os
import numpy as np
from numpy.linalg import inv
from datetime import datetime
import time
import math
import copy
#import pcl
import open3d as o3d
import cv2

import rospy
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2

from scipy.spatial.transform import Rotation as R
#import tf.transformations as tr
import time
import matplotlib.pyplot as plt

#plt.switch_backend('agg')

import torch
import torch.nn as nn
from model import MotionNet, MotionNetMGDA, FeatEncoder

global init, pcd_old, tf_curr
init = True
tf_curr = np.identity(4)

trained_model_path = '/mnt/Disk2/download/model_MGDA.pth'
img_save_dir = '/home/joinet/MotionNet/logs/itri/tmp'

save_directory = '/mnt/Disk1/itri_global_vel_i2o1_no_mask'

history_scan_num = 2
out_seq_len = 1

voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
use_adj_frame_pred = True
use_motion_state_pred_masking = True
color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

pc_list = []
tf_list = []

lidar_radar_ts_bias = 148.9810913 - 0.19 #1248.675218 # lidar_ts - radar_ts # TODO: this should be calibrated

lidar_buffer = []
lidar_ts_buffer = []
res_buffer = []
pose_buffer = []

radar_buffer = []
radar_ts_buffer = []

def pc_msg_to_np(pc_msg):
    gen = point_cloud2.read_points(pc_msg)
    pointcloud = []
    for p in gen:
        point = [p[0],p[1],p[2],p[3]]
        pointcloud.append(point)
    np_pointcloud = np.array(pointcloud)

    return np_pointcloud

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source.paint_uniform_color([0, 1, 0])
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp, source])

def trans_pc(pc_list, tf_list):
    len_ = len(tf_list)
    local_tf_list = []
    for i in range(len_):
      local_lidar_tf = np.dot(np.linalg.inv(tf_list[len_-1]), tf_list[i])
      local_tf_list.append(local_lidar_tf)
    trans_pc_list = []
    for i in range(len_):
      trans_pc_list.append( np.dot(local_tf_list[i], pc_list[i].T).T )

    return trans_pc_list
def trans_single_pc(pc, mat):
  hom_pc = np.concatenate((pc, np.ones((pc.shape[0],1))), axis=1)
  hom_pc_ = np.dot(mat, hom_pc.T).T
  #p1 = o3d.geometry.PointCloud()
  #p1.points = o3d.utility.Vector3dVector(hom_pc[:,:-1])
  #p2 = o3d.geometry.PointCloud()
  #p2.points = o3d.utility.Vector3dVector(hom_pc_[:,:-1])
  #p1.paint_uniform_color([0, 0, 1])
  #p2.paint_uniform_color([1, 0, 0])
  #o3d.visualization.draw_geometries([p1, p2])
  return hom_pc_[:,:-1]
def voxelize_occupy(pts, voxel_size, extents=None, return_indices=False):
    """
    Voxelize the input point cloud. We only record if a given voxel is occupied or not, which is just binary indicator.

    The input for the voxelization is expected to be a PointCloud
    with N points in 4 dimension (x,y,z,i). Voxel size is the quantization size for the voxel grid.

    voxel_size: I.e. if voxel size is 1 m, the voxel space will be
    divided up within 1m x 1m x 1m space. This space will be 0 if free/occluded and 1 otherwise.
    min_voxel_coord: coordinates of the minimum on each axis for the voxel grid
    max_voxel_coord: coordinates of the maximum on each axis for the voxel grid
    num_divisions: number of grids in each axis
    leaf_layout: the voxel grid of size (numDivisions) that contain -1 for free, 0 for occupied

    :param pts: Point cloud as N x [x, y, z, i]
    :param voxel_size: Quantization size for the grid, vd, vh, vw
    :param extents: Optional, specifies the full extents of the point cloud.
                    Used for creating same sized voxel grids. Shape (3, 2)
    :param return_indices: Whether to return the non-empty voxel indices.
    """
    # Function Constants
    VOXEL_EMPTY = 0
    VOXEL_FILLED = 1

    # Check if points are 3D, otherwise early exit
    if pts.shape[1] < 3 or pts.shape[1] > 4:
        raise ValueError("Points have the wrong shape: {}".format(pts.shape))

    if extents is not None:
        if extents.shape != (3, 2):
            raise ValueError("Extents are the wrong shape {}".format(extents.shape))

        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]

    # Discretize voxel coordinates to given quantization size
    discrete_pts = np.floor(pts[:, :3] / voxel_size).astype(np.int32)

    # Use Lex Sort, sort by x, then y, then z
    x_col = discrete_pts[:, 0]
    y_col = discrete_pts[:, 1]
    z_col = discrete_pts[:, 2]
    sorted_order = np.lexsort((z_col, y_col, x_col))

    # Save original points in sorted order
    discrete_pts = discrete_pts[sorted_order]

    # Format the array to c-contiguous array for unique function
    contiguous_array = np.ascontiguousarray(discrete_pts).view(
        np.dtype((np.void, discrete_pts.dtype.itemsize * discrete_pts.shape[1])))

    # The new coordinates are the discretized array with its unique indexes
    _, unique_indices = np.unique(contiguous_array, return_index=True)

    # Sort unique indices to preserve order
    unique_indices.sort()

    voxel_coords = discrete_pts[unique_indices]

    # Compute the minimum and maximum voxel coordinates
    if extents is not None:
        min_voxel_coord = np.floor(extents.T[0] / voxel_size)
        max_voxel_coord = np.ceil(extents.T[1] / voxel_size) - 1
    else:
        min_voxel_coord = np.amin(voxel_coords, axis=0)
        max_voxel_coord = np.amax(voxel_coords, axis=0)

    # Get the voxel grid dimensions
    num_divisions = ((max_voxel_coord - min_voxel_coord) + 1).astype(np.int32)

    # Bring the min voxel to the origin
    voxel_indices = (voxel_coords - min_voxel_coord).astype(int)

    # Create Voxel Object with -1 as empty/occluded
    leaf_layout = VOXEL_EMPTY * np.ones(num_divisions.astype(int), dtype=np.float32)

    # Fill out the leaf layout
    leaf_layout[voxel_indices[:, 0],
                voxel_indices[:, 1],
                voxel_indices[:, 2]] = VOXEL_FILLED

    if return_indices:
        return leaf_layout, voxel_indices
    else:
        return leaf_layout


def lidar_res_plot(viz_map, cat_pred, motion_pred_numpy, disp_pred, non_empty_map, ax):
    border_meter = 4
    border_pixel = border_meter * 4
    x_lim = [-(32 - border_meter), (32 - border_meter)]
    y_lim = [-(32 - border_meter), (32 - border_meter)]

    cat_pred = np.argmax(cat_pred, axis=0) + 1
    #cat_pred = (cat_pred * non_empty_map * filter_mask).astype(np.int)
    cat_pred = (cat_pred * non_empty_map).astype(np.int)

    idx_x = np.arange(256)
    idx_y = np.arange(256)
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    qk = [None] * len(color_map)  # for quiver key

    ax[0].imshow(np.rot90(viz_map), cmap = 'jet') # non_empty_map binary
    ax[0].axis('off')
    ax[0].set_aspect('equal')
    ax[0].title.set_text('LIDAR data')

    motion_pred_numpy = np.squeeze(motion_pred_numpy) * non_empty_map

    ax[1].imshow(np.rot90(motion_pred_numpy), cmap = 'gray')
    ax[1].axis('off')
    ax[1].set_aspect('equal')
    ax[1].title.set_text('motion segmentation')
    '''
    vel_map = np.sqrt(np.power(disp_pred[19,:,:,0],2)+np.power(disp_pred[19,:,:,1],2)) * non_empty_map
    upper_bound_mask = vel_map >= 20
    vel_map[upper_bound_mask] = 20

    ax[1,1].imshow(np.rot90(vel_map), cmap = 'jet', vmin=0, vmax=10)
    ax[1,1].axis('off')
    ax[1,1].set_aspect('equal')
    ax[1,1].title.set_text('vel_map')

    binary_vel_map = np.ones_like(vel_map)
    thd_mask = vel_map < 0.5
    binary_vel_map[thd_mask] = 0
    ax[1,2].imshow(np.rot90(binary_vel_map), cmap = 'gray')
    ax[1,2].axis('off')
    ax[1,2].set_aspect('equal')
    ax[1,2].title.set_text('vel_map')
    '''

    for k in range(len(color_map)):
      # ------------------------ Prediction ------------------------
      # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
      mask_pred = cat_pred == (k + 1)
      field_pred = disp_pred[-1]  # Show last prediction, ie., the 20-th frame

      # For cells with very small movements, we threshold them to be static
      field_pred_norm = np.linalg.norm(field_pred, ord=2, axis=-1)  # out: (h, w)
      thd_mask = field_pred_norm <= 0.4
      field_pred[thd_mask, :] = 0

      # We use the same indices as the ground-truth, since we are currently focused on the foreground
      X_pred = idx_x[mask_pred]
      Y_pred = idx_y[mask_pred]
      U_pred = field_pred[:, :, 0][mask_pred] / voxel_size[0]
      V_pred = field_pred[:, :, 1][mask_pred] / voxel_size[1]

      qk[k] = ax[2].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])
      ax[2].quiverkey(qk[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
      ax[2].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, width = 0.003, color=color_map[k])
      ax[2].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
      ax[2].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
      ax[2].set_aspect('equal')
      ax[2].title.set_text('Prediction')
      ax[2].axis('off')


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
def gen_gt_vel(disp_list):
  combined_polar_lidar = combined_lidar_scans(disp_list)
  result_cart_radar = img_polar_to_cart(combined_polar_lidar)
  return result_cart_radar
def gen_gt_mask(rmg_data_list, radar_cart):
  MASKING=False
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

def viz_err_fig(raw_radar, gt, res):
  viz_err = np.zeros((256,256,3))
  viz_err[:,:,0] = gt * np.logical_not(res)
  viz_err[:,:,1] = gt * res
  viz_err[:,:,2] = res * np.logical_not(gt)
  viz_radar = np.stack((raw_radar,raw_radar,raw_radar), axis=2)
  return (viz_err + viz_radar*3)/2.

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



def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat

def gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg):
  N=256
  cart_resolution = 0.25
  static_disp_map_list = list()

  field_gt = np.zeros((N,N,2))
  field_gt[:,:,0] = result_cart_radar_disp_0
  field_gt[:,:,1] = result_cart_radar_disp_1
  mask = result_cart_radar_rmg.astype(np.bool) * result_cart_radar_seg.astype(np.bool)
  #mask = np.zeros((256,256))
  field_gt = field_gt*mask.reshape((N,N,1))

  assert len(tf_mat_list)>out_seq_len
  for idx in range(out_seq_len):
    tf_mat = tf_mat_list[idx+1] # top: x, right: y

    # transform ego-motion to pixel unit and w.r.t disp frame
    print('delta (m) :', tf_mat[0,3], tf_mat[1,3], -np.arctan(tf_mat[1,0] / tf_mat[0,0]))
    delta_x = -tf_mat[1,3]/cart_resolution
    delta_y = -tf_mat[0,3]/cart_resolution
    delta_yaw = -np.arctan(tf_mat[1,0] / tf_mat[0,0])

    delta_tf = GetMatFromXYYaw(delta_x, delta_y, delta_yaw) # w.r.t disp frame
    #print('delta:', delta_x, delta_y, delta_yaw)

    valid_mask = result_cart_radar_rmg.astype(np.bool) # only compute disp for valid pixel
    static_disp_map = np.zeros((N,N,2))
    for i in range(N): # row
      for j in range(N): # col
        if valid_mask[i,j] == 1:
          center=N/2-0.5
          local_tf_curr_t = GetMatFromXYYaw(j-center, center-i, 0) #GetMatFromXYYaw(j-center, center-i, 0)
          disp_wrt_curr_t = field_gt[i,j,:]
          #print(disp_wrt_curr_t)
          disp_tf_wrt_curr_t = GetMatFromXYYaw(-disp_wrt_curr_t[1], -disp_wrt_curr_t[0], 0)
          local_tf_old_t = delta_tf.dot(inv(disp_tf_wrt_curr_t.dot(inv(local_tf_curr_t))))

          #local_tf_old_t = np.dot(delta_tf, local_tf_curr_t)
          x_mo = local_tf_old_t[0,2] - local_tf_curr_t[0,2]
          y_mo = local_tf_old_t[1,2] - local_tf_curr_t[1,2]
          static_disp_map[i,j,0] = x_mo # x-axis of disp frame
          static_disp_map[i,j,1] = y_mo # y-axis of disp frame
    static_disp_map_list.append(static_disp_map*result_cart_radar_rmg.astype(np.bool).reshape((N,N,1)))
  return static_disp_map_list

def callback_lidar(cloud):
  global init, pcd_old, tf_curr
  rospy.loginfo("Lidar %s: %s", cloud.header.seq, cloud.header.stamp.to_sec() - lidar_radar_ts_bias)
  pc = pc_msg_to_np(cloud)
  pc = pc[:,:-1] # rm intensity info
  #print(pc.shape)

  # x y z y p r
  # -0.75 0.1 0 0.06 0 0
  # -0.95 0.0 -1 0.055 0 0
  mat = np.identity(4) # lidar2radar
  r = R.from_euler('z', 0.055, degrees=False)
  mat[:3,:3] = r.as_matrix()
  mat[0,3] = -0.8 #-0.95
  mat[1,3] = 0.0
  mat = np.linalg.inv(mat) # radar2lidar
  pc = trans_single_pc(pc, mat) # trans pc to radar frame

  # rmg
  z_low_idx = np.where(pc[:,2]<-1.3)[0]
  pc_rmg = np.delete(pc, z_low_idx, axis=0)

  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(pc_rmg)

  #hom_pc = np.concatenate((pc, np.ones((pc.shape[0],1))), axis=1)
  #lidar_buffer.append(hom_pc)
  #lidar_ts_buffer.append(cloud.header.stamp.to_sec() - lidar_radar_ts_bias)

  if init:
    pcd_old = copy.deepcopy(pcd)
    init = False

  ### icp ###
  downpcd = pcd.voxel_down_sample(voxel_size=0.3) #0.2
  downpcd_old = pcd_old.voxel_down_sample(voxel_size=0.3)
  threshold = 2
  trans_init = np.identity(4)
  reg_p2p = o3d.registration.registration_icp(
          downpcd, downpcd_old, threshold, trans_init,
          o3d.registration.TransformationEstimationPointToPoint(), o3d.registration.ICPConvergenceCriteria(max_iteration = 300))
  tf_curr = np.dot(tf_curr, reg_p2p.transformation)
  #draw_registration_result(downpcd, downpcd_old, reg_p2p.transformation)
  #print(tf_curr)

  hom_pc = np.concatenate((pc, np.ones((pc.shape[0],1))), axis=1)
  hom_pc_rmg = np.concatenate((pc_rmg, np.ones((pc_rmg.shape[0],1))), axis=1)
  pc_list.append(hom_pc)
  tf_list.append(tf_curr)

  if len(pc_list) > 5 and len(tf_list) > 5:
    pc_list.pop(0)
    tf_list.pop(0)
    trans_pc_list = trans_pc(pc_list, tf_list)

    voxel_indices_list = list()
    padded_voxel_points_list = list()
    for i in range(5):
      res, voxel_indices = voxelize_occupy(trans_pc_list[i], voxel_size=voxel_size, extents=area_extents, return_indices=True)
      voxel_indices_list.append(voxel_indices)
      padded_voxel_points_list.append(res)

    padded_voxel_points = np.stack(padded_voxel_points_list, axis=0).astype(np.bool) # Shape: (5, 256, 256, 13)
    non_empty_map = padded_voxel_points[-1,:,:,:].any(axis=2)

    viz_map = np.zeros(padded_voxel_points[-1,:,:,0].shape)
    occ_map = np.ones(padded_voxel_points[-1,:,:,0].shape)
    for i in range(padded_voxel_points[-1,:,:,:].shape[2]):
      k = padded_voxel_points[-1,:,:,:].shape[2]-1-i
      viz_map = viz_map + padded_voxel_points[-1,:,:,k]*(k+1) * occ_map
      occ_map = np.logical_not(padded_voxel_points[-1,:,:,-i-1:].any(axis=2))

    padded_voxel_points = padded_voxel_points.astype(np.float32)
    padded_voxel_points = torch.from_numpy(padded_voxel_points)
    padded_voxel_points = torch.unsqueeze(padded_voxel_points, 0).to(device)
    model_encoder.eval()
    model_head.eval()
    with torch.no_grad():
      shared_feats = model_encoder(padded_voxel_points)
      disp_pred, cat_pred, motion_pred = model_head(shared_feats)

      disp_pred = disp_pred.cpu().numpy()
      disp_pred = np.transpose(disp_pred, (0, 2, 3, 1))
      cat_pred = np.squeeze(cat_pred.cpu().numpy(), 0)

      if use_adj_frame_pred:  # The prediction are the displacement between adjacent frames
          for c in range(1, disp_pred.shape[0]):
              disp_pred[c, ...] = disp_pred[c, ...] + disp_pred[c - 1, ...]

      if use_motion_state_pred_masking:
          motion_pred_numpy = motion_pred.cpu().numpy()
          motion_pred_numpy = np.argmax(motion_pred_numpy, axis=1)
          motion_mask = motion_pred_numpy == 0 # index of static pixel

          cat_pred_numpy = np.argmax(cat_pred, axis=0)
          cat_mask = np.logical_and(cat_pred_numpy == 0, non_empty_map == 1)
          cat_mask = np.expand_dims(cat_mask, 0)

          cat_weight_map = np.ones_like(motion_pred_numpy, dtype=np.float32)
          cat_weight_map[motion_mask] = 0.0
          cat_weight_map[cat_mask] = 0.0
          cat_weight_map = cat_weight_map[:, :, :, np.newaxis]  # (1, h, w. 1)

          disp_pred = disp_pred * cat_weight_map

    lidar_buffer.append(hom_pc_rmg)
    lidar_ts_buffer.append(cloud.header.stamp.to_sec() - lidar_radar_ts_bias)

    res_dict = dict()
    res_dict['disp_pred'] = disp_pred[-1]
    res_dict['cat_pred'] = cat_pred
    res_dict['motion_pred'] = motion_pred_numpy
    res_dict['non_empty_map'] = non_empty_map
    res_buffer.append(res_dict)

    pose_buffer.append(tf_curr)

    if len(lidar_ts_buffer) > 30:
      lidar_ts_buffer.pop(0)
      lidar_buffer.pop(0)
      res_buffer.pop(0)
      pose_buffer.pop(0)

#    disp_pred = np.rot90(res_dict['disp_pred'],2)
#    non_empty_map = np.rot90(res_dict['non_empty_map'],2)
#    motion_pred = np.rot90(res_dict['motion_pred'],2)

#    vel_map = np.sqrt(np.power(disp_pred[:,:,0],2)+np.power(disp_pred[:,:,1],2)) * non_empty_map
#    binary_vel_map = np.ones_like(vel_map)
#    thd_mask = vel_map < 0.01 # 0.5
#    binary_vel_map[thd_mask] = 0
#    print(binary_vel_map.shape)
#    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#    ax.imshow(binary_vel_map)
#    field_gt = np.zeros((256,256,2))
#    field_gt[:,:,0] = disp_pred[:,:,1]
#    field_gt[:,:,1] = disp_pred[:,:,0]
#    # viz same as oxford ramnet
#    idx_x = np.arange(field_gt.shape[0])
#    idx_y = np.arange(field_gt.shape[1])
#    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
#    mask = non_empty_map.astype(np.bool) #  * combined_cart_rmg.astype(np.bool) is better !!
#    X = idx_x[mask]
#    Y = idx_y[mask]
#    U = field_gt[:, :, 0][mask]
#    V = field_gt[:, :, 1][mask]
#    qk = ax.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.2, width=0.0015, headwidth=10, color='r', minlength=3, minshaft=1)
#    plt.show()

    # Plot
    #fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    #lidar_res_plot(viz_map, cat_pred, motion_pred_numpy, disp_pred, non_empty_map, ax)
    #plt.savefig(os.path.join(img_save_dir, 'res_'+str(cloud.header.stamp) + '.png'))
    #plt.show()
    #ax[0].clear()
    #ax[1].clear()
    #ax[2].clear()

  pcd_old = copy.deepcopy(pcd)

def callback_radar(radar_msg):
  global fig2, ax2
  rospy.loginfo("I heard Radar %s: %s", radar_msg.header.seq, radar_msg.header.stamp.to_sec())

  radar_img = np.frombuffer(radar_msg.data, dtype=np.uint8).reshape(radar_msg.height, radar_msg.width, -1)[...,0]
  radar_img = radar_img.astype(np.float32)/255.
#  ax2.imshow(radar_img, cmap='gray')
#  plt.pause(0.1)
#  ax2.clear()

  radar_buffer.append(radar_img)
  radar_ts_buffer.append(radar_msg.header.stamp.to_sec())

  history_scan_num=2

  if len(radar_buffer) > history_scan_num+2:
    curr_radar_img = radar_buffer[-2]
    curr_radar_ts = radar_ts_buffer[-2]
    print('-----RUN-----')
    print(curr_radar_ts)
    print('len(lidar_ts_buffer)',len(lidar_ts_buffer))
    print('len(lidar_buffer)',len(lidar_buffer))
    print('len(res_buffer)',len(res_buffer))
    print('len(pose_buffer)',len(pose_buffer))
    if len(lidar_ts_buffer)<3:
      print('no sync lidar result yet...')
      return
    sync_lidar_idx, sync_lidar_ts = get_sync(curr_radar_ts, np.array(lidar_ts_buffer))
    while(np.abs(sync_lidar_ts-curr_radar_ts)>0.1 or sync_lidar_idx>len(lidar_ts_buffer)-3):
      print('wait for lidar processing...')
      time.sleep(0.55)
      sync_lidar_idx, sync_lidar_ts = get_sync(curr_radar_ts, np.array(lidar_ts_buffer))
    print('len(lidar_ts_buffer)',len(lidar_ts_buffer))
    print(sync_lidar_idx, sync_lidar_ts, 't diff:', np.abs(sync_lidar_ts-curr_radar_ts))

    rmg_data_list = list()
    voxel_size = (0.25, 0.25, 10.0)
    area_extents = np.array([[-32., 32.], [-32., 32.], [-10., 10.]])
    for i in range(5):
      padded_voxel_points, voxel_indices = voxelize_occupy(lidar_buffer[sync_lidar_idx+i], voxel_size=voxel_size, extents=area_extents, return_indices=True)
      non_empty_map = padded_voxel_points[:,:,:].any(axis=2).astype(np.float32)
      non_empty_map = np.rot90(non_empty_map,2)
      rmg_data_list.append(non_empty_map)

    motion_seg_list = list()
    disp_0_list = list()
    disp_1_list = list()
    for i in range(5):
      # Read res dict
      res_data = res_buffer[sync_lidar_idx+i]
      disp_pred = res_data['disp_pred']
      non_empty_map = res_data['non_empty_map']
      motion_pred = res_data['motion_pred']

      # Coordinate transform
      disp_pred = np.rot90(disp_pred,2)
      non_empty_map = np.rot90(non_empty_map,2)
      motion_pred = np.rot90(motion_pred,2)

      # Pre-process
      vel_map = np.sqrt(np.power(disp_pred[:,:,0],2)+np.power(disp_pred[:,:,1],2)) * non_empty_map
      binary_vel_map = np.ones_like(vel_map)
      thd_mask = vel_map < 0.01 # 0.5
      binary_vel_map[thd_mask] = 0
      #print(binary_vel_map.shape)

#      fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#      ax.imshow(binary_vel_map)
#      field_gt = np.zeros((256,256,2))
#      field_gt[:,:,0] = disp_pred[:,:,1]
#      field_gt[:,:,1] = disp_pred[:,:,0]
#      # viz same as oxford ramnet
#      idx_x = np.arange(field_gt.shape[0])
#      idx_y = np.arange(field_gt.shape[1])
#      idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
#      mask = non_empty_map.astype(np.bool) #  * combined_cart_rmg.astype(np.bool) is better !!
#      X = idx_x[mask]
#      Y = idx_y[mask]
#      U = field_gt[:, :, 0][mask]
#      V = field_gt[:, :, 1][mask]
#      qk = ax.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.2, width=0.0015, headwidth=10, color='r', minlength=3, minshaft=1)
#      plt.show()

      motion_seg_list.append(binary_vel_map)
      disp_0_list.append(disp_pred[:,:,0])
      disp_1_list.append(disp_pred[:,:,1])

    tf_mat_list = list()
    curr_pose = pose_buffer[sync_lidar_idx]
    for i in range(history_scan_num):
      old_pose = pose_buffer[sync_lidar_idx-i]
      tf_mat = np.linalg.inv(old_pose).dot(curr_pose)
      tf_mat_list.append(tf_mat)

    result_cart_radar_rmg, combined_cart_rmg = gen_gt_mask(rmg_data_list, radar_img)
    result_cart_radar_seg, combined_cart_seg = gen_gt_mask(motion_seg_list, radar_img)
    result_cart_radar_disp_0 = gen_gt_vel(disp_0_list)
    result_cart_radar_disp_1 = gen_gt_vel(disp_1_list)

    ### Generate disp by motion ###
    #result_cart_radar_disp_0 = np.zeros((256,256))
    #result_cart_radar_disp_1 = np.zeros((256,256))
    #result_disp_map_seq_list = gen_static_disp_map(radar_buffer, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg)

    save_data_dict = dict()
    for i in range(history_scan_num):
      save_data_dict['raw_radar_' + str(i)] = radar_buffer[-2-i]
    save_data_dict['gt_disp_global'] = np.concatenate((np.expand_dims(result_cart_radar_disp_0,axis=2), np.expand_dims(result_cart_radar_disp_1,axis=2)), axis=2)
    save_data_dict['gt_radar'] = result_cart_radar_rmg
    save_data_dict['gt_moving'] = result_cart_radar_seg
    save_data_dict['gt_lidar'] = combined_cart_rmg

    save_file_name = os.path.join(save_directory, str(curr_radar_ts) + '.npy')
    #np.save(save_file_name, arr=save_data_dict)

    if len(radar_ts_buffer) > 10:
      radar_ts_buffer.pop(0)
      radar_buffer.pop(0)

#    fig, ax = plt.subplots(1, 3, figsize=(14, 7))

#    viz = viz_err_fig(curr_radar_img, result_cart_radar_seg, result_cart_radar_rmg)

#    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
#    ax.imshow(result_cart_radar_seg)
#    field_gt = np.zeros((256,256,2))
#    field_gt[:,:,0] = result_cart_radar_disp_1
#    field_gt[:,:,1] = result_cart_radar_disp_0
#    # viz same as oxford ramnet
#    idx_x = np.arange(field_gt.shape[0])
#    idx_y = np.arange(field_gt.shape[1])
#    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
#    mask = non_empty_map.astype(np.bool) #  * combined_cart_rmg.astype(np.bool) is better !!
#    X = idx_x[mask]
#    Y = idx_y[mask]
#    U = field_gt[:, :, 0][mask]
#    V = field_gt[:, :, 1][mask]
#    qk = ax.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.2, width=0.0015, headwidth=10, color='r', minlength=3, minshaft=1)
#    plt.show()

    #plt.pause(0.1)
    #plt.savefig(os.path.join(img_save_dir, str(radar_msg.header.stamp.to_sec()) + '.png'))
    #ax2.clear()


def listener():
    global fig, ax, fig2, ax2

    #fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    #fig2, ax2 = plt.subplots(1, 1, figsize=(4, 4))
    #plt.tight_layout()

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/velodyne_points", PointCloud2, callback_lidar, queue_size=100000)
    rospy.Subscriber("/Navtech/Cartesian", Image, callback_radar, queue_size=100000)
    rospy.spin()


if __name__ == '__main__':
    ### Load Model ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model_encoder = FeatEncoder()
    model_head = MotionNetMGDA(out_seq_len=20, motion_category_num=2)
    model_encoder = nn.DataParallel(model_encoder)
    model_head = nn.DataParallel(model_head)
    checkpoint = torch.load(trained_model_path)
    model_encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model_head.load_state_dict(checkpoint['head_state_dict'])
    model_encoder = model_encoder.to(device)
    model_head = model_head.to(device)
    loaded_models = [model_encoder, model_head]
    print("Loaded pretrained model...")
    model_encoder = loaded_models[0]
    model_head = loaded_models[1]
    listener()
