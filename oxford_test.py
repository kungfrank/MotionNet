# Run with python2 for rviz debuging

import argparse
import pandas as pd
import time
import os
import numpy as np
from datetime import datetime
import time
import math

### Maybe this could make cv2 and ros work together in python3 ? ###
#import cv2
#from envbash import load_envbash
#load_envbash('/opt/ros/kinetic/setup.bash', override=True)
#import sys
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

#import rospy
#from std_msgs.msg import Header
#from sensor_msgs import point_cloud2
#from sensor_msgs.msg import PointCloud2, PointField

from scipy.spatial.transform import Rotation as R
#import tf.transformations as tr
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from model import MotionNet, MotionNetMGDA, FeatEncoder


# pseudo_lidar_to_lidar_left_tf
pseudo_lidar_to_lidar_left_tf = np.zeros((4,4))
r = R.from_quat([-0.704, 0.710, 0.017, 0.013])
pseudo_lidar_to_lidar_left_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_lidar_left_tf[:3,3] = np.array([-0.449, 0.011, -0.001])
pseudo_lidar_to_lidar_left_tf[3,3] = 1

# pseudo_lidar_to_lidar_right_tf
pseudo_lidar_to_lidar_right_tf = np.zeros((4,4))
r = R.from_quat([-0.705, 0.709, 0.014, 0.016])
pseudo_lidar_to_lidar_right_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_lidar_right_tf[:3,3] = np.array([0.449, -0.000, 0.001])
pseudo_lidar_to_lidar_right_tf[3,3] = 1

# pseudo_lidar_to_ins_tf
pseudo_lidar_to_ins_tf = np.zeros((4,4))
r = R.from_quat([0.709, 0.705, 0.019, -0.010])
pseudo_lidar_to_ins_tf[:3,:3] = r.as_matrix()
pseudo_lidar_to_ins_tf[:3,3] = np.array([0.010, -1.102, -1.464])
pseudo_lidar_to_ins_tf[3,3] = 1

pc_list = []
tf_list = []

# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence

trained_model_path = '/mnt/Disk2/download/model_MGDA.pth'
img_save_dir = '/home/joinet/MotionNet/logs/oxford/'
#data_save_dir = '/mnt/Disk1/oxford_res/'
data_save_dir = '/mnt/Disk1/oxford_res_combined_ts_from_pseudo_lidar_test/2019-01-10-11-46-21-radar-oxford-10k/'

use_adj_frame_pred = True
use_motion_state_pred_masking = True
color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

import os
pid = os.getpid()
print('pid:', pid)

def UnixTimeToSec(unix_timestamp):
    #print('Unix timestamp:')
    #print(str(unix_timestamp))
    time = datetime.fromtimestamp(unix_timestamp / 1000000)
    s = unix_timestamp % 1000000
    sec_timestamp = time.hour*3600 + time.minute*60 + time.second + (float(s)/1000000)
    #print("timestamp: ")
    #print(sec_timestamp)
    return sec_timestamp

def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat

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

def read_point_cloud(path):
    return np.fromfile(path, dtype=np.float32).reshape(4, -1) # (x,y,z,I) * N


def gen_local_map(pc_list, tf_list):
  len_ = len(tf_list)
  local_tf_list = []
  for i in range(len_):
    local_ins_tf = np.dot(np.linalg.inv(tf_list[len_-1]), tf_list[i])
    local_lidar_tf = np.dot(np.dot(pseudo_lidar_to_ins_tf, local_ins_tf), np.linalg.inv(pseudo_lidar_to_ins_tf))
    local_tf_list.append(local_lidar_tf)

  local_map_pc = np.empty([1, 4])
  for i in range(len_):
    trans_pc = np.dot(local_tf_list[i], pc_list[i].T).T
    local_map_pc = np.concatenate((local_map_pc, trans_pc), axis=0)

  return local_map_pc[:-1]


def trans_pc(pc_list, tf_list):
  len_ = len(tf_list)
  local_tf_list = []
  for i in range(len_):
    local_ins_tf = np.dot(np.linalg.inv(tf_list[len_-1]), tf_list[i])
    local_lidar_tf = np.dot(np.dot(pseudo_lidar_to_ins_tf, local_ins_tf), np.linalg.inv(pseudo_lidar_to_ins_tf))
    local_tf_list.append(local_lidar_tf)

  trans_pc_list = []
  for i in range(len_):
    trans_pc_list.append( np.dot(local_tf_list[i], pc_list[i].T).T )

  return trans_pc_list

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


def main():
    ### ROS ###
#    rospy.init_node('talker', anonymous=True)
#    pub = rospy.Publisher('pc', PointCloud2, queue_size=100)
#    lidar_pub = rospy.Publisher('lidar_pc', PointCloud2, queue_size=100)

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

    #fig, ax = plt.subplots(1, 3, figsize=(20, 8))
    fig, ax = plt.subplots(2, 3, figsize=(20, 14))
    plt.tight_layout()

    for filename in os.listdir("/mnt/Disk2/Oxford"):
      if filename != '2019-01-10-11-46-21-radar-oxford-10k': # '2019-01-10-11-46-21-radar-oxford-10k' '2019-01-10-12-32-52-radar-oxford-10k'
        continue
      print(filename)

      path = '/mnt/Disk2/Oxford/' + filename + '/'
      radar_folder = path+'radar/'
      timestamps_path = path+'radar.timestamps'
      radar_timestamps = np.loadtxt(timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

      gps_file = path+'/gps/ins.csv'
      gps_data = pd.read_csv(gps_file)
      gps_timestamps = gps_data.iloc[:,0].copy()

      lidar_path = '/mnt/Disk2/Oxford_lidar/' + filename + '/'
      lidar_l_folder = lidar_path+'velodyne_left/'
      lidar_l_timestamps_path = lidar_path+'velodyne_left.timestamps'
      lidar_l_timestamps = np.loadtxt(lidar_l_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

      lidar_r_folder = lidar_path+'velodyne_right/'
      lidar_r_timestamps_path = lidar_path+'velodyne_right.timestamps'
      lidar_r_timestamps = np.loadtxt(lidar_r_timestamps_path, delimiter=' ', usecols=[0], dtype=np.int64)

      # get first and last radar timestamp #
      radar_init_t = UnixTimeToSec(radar_timestamps[0])
      radar_end_t = UnixTimeToSec(radar_timestamps[np.size(radar_timestamps)-1])
      print("radar_init_t:")
      print(radar_init_t)
      print("radar_end_t:")
      print(radar_end_t)

      frame_idx = 0
      for lidar_l_timestamp in lidar_l_timestamps:
        frame_idx = frame_idx + 1
        if UnixTimeToSec(lidar_l_timestamp) > radar_init_t and UnixTimeToSec(lidar_l_timestamp) <= radar_end_t :

          lidar_r_idx, lidar_r_timestamp = get_sync(lidar_l_timestamp, lidar_r_timestamps)
          gps_idx, gps_timestamp = get_sync(lidar_l_timestamp, gps_timestamps)

          curr_lidar_l_t = UnixTimeToSec(lidar_l_timestamp)
          curr_lidar_r_t = UnixTimeToSec(lidar_r_timestamp)
          curr_gps_t = UnixTimeToSec(gps_timestamp)

          lidar_l_filename = lidar_l_folder + str(lidar_l_timestamp) + '.bin'
          if not os.path.isfile(lidar_l_filename):
              print("Could not find radar example: {}".format(lidar_l_filename))
          pc_left = read_point_cloud(lidar_l_filename).T # (N x 4)
          pc_left[:,3] = 1
          new_pc_left = np.dot(pseudo_lidar_to_lidar_left_tf, pc_left.T).T

          lidar_r_filename = lidar_r_folder + str(lidar_r_timestamp) + '.bin'
          if not os.path.isfile(lidar_r_filename):
              print("Could not find radar example: {}".format(lidar_r_filename))
          pc_right = read_point_cloud(lidar_r_filename).T # (N x 4)
          pc_right[:,3] = 1
          new_pc_right = np.dot(pseudo_lidar_to_lidar_right_tf, pc_right.T).T

          new_pc = np.concatenate((new_pc_left, new_pc_right), axis=0)

#          header = Header()
#          header.stamp = rospy.get_rostime() #rospy.Time.from_sec(curr_t)
#          header.frame_id = "/pseudo_lidar"
#          fields = [PointField('x', 0, PointField.FLOAT32, 1),
#                PointField('y', 4, PointField.FLOAT32, 1),
#                PointField('z', 8, PointField.FLOAT32, 1)
#                ]
#          pc_msg = point_cloud2.create_cloud(header, fields, new_pc[:,:3])
#          lidar_pub.publish(pc_msg)

          northing = gps_data.iloc[gps_idx,5]
          easting = gps_data.iloc[gps_idx,6]
          down = gps_data.iloc[gps_idx,7]
          roll = gps_data.iloc[gps_idx,12]
          pitch = gps_data.iloc[gps_idx,13]
          yaw = gps_data.iloc[gps_idx,14]
          tf_curr = np.zeros((4,4))
          r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)
          tf_curr[:3,:3] = r.as_matrix()
          tf_curr[:3,3] = np.array([northing, easting, down])
          tf_curr[3,3] = 1
          np.set_printoptions(suppress=True)

#          q = tr.quaternion_from_euler(roll, pitch, yaw) # roll, pitch, yaw
#          tf_curr = tr.quaternion_matrix(q)
#          tf_curr[0:3, -1] = [northing, easting, down]
#          np.set_printoptions(suppress=True)
#          print(tf_curr)

          pc_list.append(new_pc)
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
            non_empty_map[125:133,121:137] = 0 #128:130 # remove car !!!

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

              ### save result ###
              save_data_dict = dict()
              save_data_dict['disp_pred'] = disp_pred[-1]
              save_data_dict['cat_pred'] = cat_pred
              save_data_dict['motion_pred'] = motion_pred_numpy
              save_data_dict['non_empty_map'] = non_empty_map
              save_data_dict['viz_map'] = viz_map

            print('=================')
            file_name = format((curr_lidar_l_t+curr_lidar_r_t)/2., '.7f') # file_name = curr_lidar_l_t
            print(file_name)
            #np.save(data_save_dir+str(file_name)+'.npy', save_data_dict)

            #f = open(data_save_dir+'res.timestamps', 'a')
            #f.write(file_name+'\n')
            #f.close()
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

            ax[0,0].imshow(np.rot90(viz_map), cmap = 'jet') # non_empty_map binary
            ax[0,0].axis('off')
            ax[0,0].set_aspect('equal')
            ax[0,0].title.set_text('LIDAR data')

            motion_pred_numpy = np.squeeze(motion_pred_numpy) * non_empty_map

            ax[0,1].imshow(np.rot90(motion_pred_numpy), cmap = 'gray')
            ax[0,1].axis('off')
            ax[0,1].set_aspect('equal')
            ax[0,1].title.set_text('motion segmentation')

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

              qk[k] = ax[0,2].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])
              ax[0,2].quiverkey(qk[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
              ax[0,2].quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, width = 0.003, color=color_map[k])
              ax[0,2].set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
              ax[0,2].set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
              ax[0,2].set_aspect('equal')
              ax[0,2].title.set_text('Prediction')
              ax[0,2].axis('off')
            plt.pause(0.1)
            #plt.savefig(os.path.join(img_save_dir, str(frame_idx) + '.png'))
            ax[0,0].clear()
            ax[0,1].clear()
            ax[0,2].clear()
            ax[1,0].clear()
            ax[1,1].clear()
            ax[1,2].clear()



if __name__ == '__main__':
  main()
