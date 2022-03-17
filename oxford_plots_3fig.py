import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
#import imageio
import argparse
import time
import open3d as o3d

#from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
#from nuscenes.nuscenes import NuScenes
#from nuscenes.utils.data_classes import LidarPointCloud
from data.data_utils import voxelize_occupy#, calc_displace_vector, point_in_hull_fast
#from model import MotionNet, MotionNetMGDA, FeatEncoder
from oxford_model import RaMNet
from scipy.spatial.transform import Rotation as R
import math
from numpy.linalg import inv
import cv2

color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

#tmp_path = '/mnt/Disk1/training_data_mo_seg_0.05_no_mask_labeled/2019-01-10-12-32-52-radar-oxford-10k_origin_'

plot_class = True
plot_motion = False
plot_flow = False

motion_comp = False

img_save_dir = '/mnt/Disk1/viz_result_3fig/2021-05-18_16-16-16_ep10_test_polar'
num_past_frames = 2
out_seq_len = 1
height_feat_size=1
plot_motion_seg = True
use_temporal_info = True

cart_resolution = 0.25

global frame_idx
frame_idx = 5

global count
count = 0

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def viz_err_fig(raw_radar, gt, res):
  viz_err = np.zeros((256,256,3))
  viz_err[:,:,0] = gt * np.logical_not(res)
  viz_err[:,:,1] = gt * res
  viz_err[:,:,2] = res * np.logical_not(gt)
  viz_radar = np.stack((raw_radar,raw_radar,raw_radar), axis=2)
  return (viz_err + viz_radar*3)/2.

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

def viz_denoise_gt(img, denoised_img, gt_img):
  viz_img = np.zeros((256,256,3))
  viz_img = np.stack((img,img,img), axis=2)

  viz_only_denoised_img = np.zeros((256,256,3))
  viz_only_denoised_img[:,:,2] = denoised_img*np.logical_not(gt_img)

  viz_only_gt_img = np.zeros((256,256,3))
  viz_only_gt_img[:,:,0] = gt_img*np.logical_not(denoised_img)

  viz_correct_img = np.zeros((256,256,3))
  viz_correct_img[:,:,1] = gt_img*denoised_img

  return (viz_img*4+viz_only_denoised_img+viz_only_gt_img+viz_correct_img)/4.



def gt_to_pixel_map_gt(radar_gt):
  pixel_radar_map = np.zeros((radar_gt.shape[0],radar_gt.shape[1],2))
  pixel_radar_map[:,:,0] = radar_gt # valid
  pixel_radar_map[:,:,1] = np.logical_not(radar_gt) # invalid
  pixel_radar_map_list = list()
  pixel_radar_map_list.append(pixel_radar_map)
  pixel_radar_map_list = np.stack(pixel_radar_map_list, 0)
  pixel_radar_map_list = torch.from_numpy(pixel_radar_map_list)
  pixel_radar_map_gt = pixel_radar_map_list.permute(0, 3, 1, 2)
  return pixel_radar_map_gt



def plot_quiver(ax_, disp0, disp1, viz_cat, color='g'):
  # Plot quiver.
  field_gt = np.zeros((256,256,2))
  field_gt[:,:,0] = disp0
  field_gt[:,:,1] = disp1
  idx_x = np.arange(field_gt.shape[0])
  idx_y = np.arange(field_gt.shape[1])
  idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
  # For cells with very small movements, we threshold them to be static
  field_gt_norm = np.linalg.norm(field_gt, ord=2, axis=-1)  # out: (h, w)
  #thd_mask = field_gt_norm <= 0.1
  #field_gt[thd_mask, :] = 0
  # Get the displacement field
  mask = viz_cat.astype(np.bool)
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  qk = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=1, width=0.0007, headwidth=20, color=color, alpha=0.6)
  #qk = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.001, headwidth=20, color=color, alpha=0.6)

  return qk


def flow_to_img(flow_x, flow_y):
  hsv = np.zeros((flow_x.shape[0],flow_x.shape[1],3)).astype(np.uint8)
  hsv[...,1] = 255
  mag, ang = cv2.cartToPolar(flow_x, flow_y)
  mag[mag>=15]=15 # upper bound
  hsv[...,0] = ang*180/np.pi/2
  #hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
  hsv[...,2] = mag/15. * 255
  rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
  return rgb.astype(np.float32)/255.

def opticalflow(prvs_,next_):
    hsv = np.zeros((prvs_.shape[0],prvs_.shape[1],3)).astype(np.uint8)
    hsv[...,1] = 255
    flow = cv2.calcOpticalFlowFarneback(prvs_,next_, None, 0.5, 3, 12, 3, 5, 1.2, 0) # 6
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return rgb.astype(np.float32)/255.


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

def calc_odom_by_disp_map(disp0, disp1, radar_mask, ransac_param):
  disp_map = np.zeros((256,256,2))
  disp_map[:,:,0] = disp0
  disp_map[:,:,1] = disp1

  radar_mask[radar_mask>0]=1
  radar_mask[radar_mask<=0]=0

  pointcloud = list()
  pointcloud_ = list()
  N=256
  center=N/2-0.5
  for i in range(N): # row
    for j in range(N): # col
      if radar_mask[i,j]==1:
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
  M, mask = cv2.findHomography(pc, pc_, cv2.RANSAC, ransac_param) # 1~10 => strict~loose 2

  # gen outlier mask
  outlier_mask = np.copy(radar_mask).astype(np.bool)
  pc_inlier = np.delete(pc, np.where(mask==0), axis=0)
  for point in pc_inlier:
    i = (center-point[0]).astype(np.int)
    j = (center+point[1]).astype(np.int)
    outlier_mask[i,j] = 0

  #print(M[1,2], M[0,2], -np.arctan(M[1,0]/M[0,0])) # w.r.t. disp frame
  np_corres_new = np_corres[mask.squeeze().astype(np.bool),:]
  corres_new = o3d.utility.Vector2iVector(np_corres_new)
  line_set = gen_corr_line_set(pc.T, pc_.T, corres_new, [0,0,1])
  #o3d.visualization.draw_geometries([pcd+pcd_]+[line_set])
  return M, outlier_mask


def find(s, ch):
  return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
  idx = int(idx)
  return idx



def GetMatFromXYYaw(x,y,yaw):
    tf_mat = np.matrix([[math.cos(yaw), -math.sin(yaw), x]
                       ,[math.sin(yaw), math.cos(yaw), y]
                       ,[0, 0, 1]])
    return tf_mat
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
    static_disp_map_list.append(static_disp_map*result_cart_radar_rmg.astype(np.bool).reshape((N,N,1)))
  return static_disp_map_list


def calc_ego_motion_disp_map(gt_dict, num_past_pcs):
  # compute ego-motion compensated disp map #
  radar_cart_list = list()
  for i in range(num_past_pcs):
    radar_cart_list.append(gt_dict['raw_radar_' + str(i)])
  tf_mat_list = list()
  for i in range(num_past_pcs):
    x, y, yaw = gt_dict['gt_tf_' + str(i)] # w.r.t. disp frame
    tf_mat = GetMatFromXYYaw(y, x, -yaw) # w.r.t. odom frame
    tf_mat_list.append(tf_mat)
  result_cart_radar_disp_0 = np.zeros((256,256))
  result_cart_radar_disp_1 = np.zeros((256,256))
  result_cart_radar_rmg = gt_dict['gt_radar']
  result_cart_radar_seg = gt_dict['gt_moving']
  result_disp_map_seq_list = gen_static_disp_map(radar_cart_list, tf_mat_list, result_cart_radar_disp_0, result_cart_radar_disp_1, out_seq_len, result_cart_radar_rmg, result_cart_radar_seg)
  ego_motion_disp_map = result_disp_map_seq_list[0]
  return ego_motion_disp_map

def opticalflow_pred(raw_radars, radar_gt, ransac_param):
  raw_radars = raw_radars.detach().numpy()
  # flow estimation
  prvs_ = (raw_radars[0,0,:,:,0]*255.).astype(np.uint8)
  next_ =  (raw_radars[0,1,:,:,0]*255.).astype(np.uint8)
  flow = cv2.calcOpticalFlowFarneback(prvs_,next_, None, 0.5, 3, 12, 3, 5, 1.2, 0) # (256, 256, 2)
  # coordinate transform
  flow[:,:,1] = -flow[:,:,1]
  # motion seg by flow
  M, outlier_mask = calc_odom_by_disp_map(flow[:,:,0], flow[:,:,1], radar_gt, ransac_param)

  return flow, outlier_mask


def calc_err(radar_gt_, motion_gt_, disp_gt_, disp_gt_global_, cat_pred_, motion_pred_, disp_pred_, device):
  global count

  # compute class error #
  class_err = torch.sum((cat_pred_-radar_gt_)**2) / (cat_pred_.shape[0]*cat_pred_.shape[1])
  class_err_avg.update(class_err.item())
  print(class_err_avg)
  #f, a = plt.subplots(1, 2, figsize=(10, 5))
  #a[0].imshow(radar_gt_.cpu().numpy())
  #a[1].imshow(power_thres.cpu().numpy())
  #plt.show()

  # compute motion error #
  motion_err = torch.sum((motion_pred_*radar_gt_-motion_gt_*radar_gt_)**2) / torch.nonzero(radar_gt_).size(0)
  valid_size = torch.nonzero(radar_gt_).size(0)
  motion_err_avg.update(motion_err.item(), valid_size)
  print(motion_err_avg)

  disp_gt_global_norm_ = torch.norm(disp_gt_global_, dim=2)
  speed_mask_0 = disp_gt_global_norm_<5
  speed_mask_1_lb = disp_gt_global_norm_>=5
  speed_mask_1_ub = disp_gt_global_norm_<10
  speed_mask_1 = speed_mask_1_lb * speed_mask_1_ub
  speed_mask_2 = disp_gt_global_norm_>10
  moving_mask_ = motion_gt_ == 1

  moving_speed0_size = torch.nonzero(speed_mask_0*moving_mask_).size(0)
  moving_speed1_size = torch.nonzero(speed_mask_1*moving_mask_).size(0)
  moving_speed2_size = torch.nonzero(speed_mask_2*moving_mask_).size(0)
  if moving_speed0_size>0:
    motion_speed0_err = torch.sum((motion_pred_*speed_mask_0*moving_mask_-motion_gt_*speed_mask_0*moving_mask_)**2) / moving_speed0_size
    motion_speed0_err_avg.update(motion_speed0_err.item(), moving_speed0_size)
  if moving_speed1_size>0:
    motion_speed1_err = torch.sum((motion_pred_*speed_mask_1*moving_mask_-motion_gt_*speed_mask_1*moving_mask_)**2) / moving_speed1_size
    motion_speed1_err_avg.update(motion_speed1_err.item(), moving_speed1_size)
  if moving_speed2_size>0:
    motion_speed2_err = torch.sum((motion_pred_*speed_mask_2*moving_mask_-motion_gt_*speed_mask_2*moving_mask_)**2) / moving_speed2_size
    motion_speed2_err_avg.update(motion_speed2_err.item(), moving_speed2_size)
  print(motion_speed0_err_avg, motion_speed1_err_avg, motion_speed2_err_avg)

  # compute disp error #

  disp_gt_ = disp_gt_global_ # !!!!!!!!!!!! for global vel est.

  disp_err_map = torch.sqrt(torch.sum((disp_gt_-disp_pred_)**2,axis=2)) # local vel est.
  valid_size = torch.nonzero(radar_gt_).size(0)
  disp_err = torch.sum(disp_err_map) / valid_size
  disp_err_avg.update(disp_err.item(), valid_size)

  # compute moving disp error #
  moving_size = torch.nonzero(radar_gt_*motion_gt_).size(0)
  if moving_size>0:
    moving_disp_err = torch.sum(disp_err_map*motion_gt_) / moving_size
    moving_disp_err_avg.update(moving_disp_err.item(), moving_size)

  # compute static disp error #
  static_gt_ = torch.logical_not(motion_gt_)
  static_size = torch.nonzero(radar_gt_*static_gt_).size(0)
  if static_size>0:
    static_disp_err = torch.sum(disp_err_map*static_gt_) / static_size
    static_disp_err_avg.update(static_disp_err.item(), static_size)
  print(disp_err_avg, moving_disp_err_avg, static_disp_err_avg)

  # compute disp error for diff disp_gt#
  disp_gt_norm_ = torch.norm(disp_gt_, dim=2) # local vel est.
  disp_mask_0 = disp_gt_norm_<5
  disp_mask_1_lb = disp_gt_norm_>=5
  disp_mask_1_ub = disp_gt_norm_<10
  disp_mask_1 = disp_mask_1_lb * disp_mask_1_ub
  disp_mask_2 = disp_gt_norm_>10

  disp0_size = torch.nonzero(disp_mask_0*radar_gt_).size(0)
  disp1_size = torch.nonzero(disp_mask_1*radar_gt_).size(0)
  disp2_size = torch.nonzero(disp_mask_2*radar_gt_).size(0)

  if disp0_size>0:
    disp0_err = torch.sum(disp_err_map*disp_mask_0) / disp0_size
    disp0_err_avg.update(disp0_err.item(), disp0_size)
  if disp1_size>0:
    disp1_err = torch.sum(disp_err_map*disp_mask_1) / disp1_size
    disp1_err_avg.update(disp1_err.item(), disp1_size)
  if disp2_size>0:
    disp2_err = torch.sum(disp_err_map*disp_mask_2) / disp2_size
    disp2_err_avg.update(disp2_err.item(), disp2_size)
  print(disp0_err_avg, disp1_err_avg, disp2_err_avg)



def img_polar_to_cart(img):
  img = np.concatenate((img[int(img.shape[0]/4):,:], img[:int(img.shape[0]/4),:]), axis=0)
  value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
  cart_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
  cart_image = cv2.resize(cart_image, (256, 256), interpolation=cv2.INTER_LINEAR)
  return cart_image

def vis_result(data_path, trained_model_path, which_model, disp, opt, save, ax1=-1, ax2=-1, ax3=-1):
  global frame_idx, count

  data_dirs = [os.path.join(data_path, f)for f in os.listdir(data_path)
                   if os.path.isfile(os.path.join(data_path, f))]
  data_dirs.sort(key=file_to_id)
  print(len(data_dirs))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  if not opt:
    # Load pre-trained network weights
    loaded_models = list()
    if which_model == "RaMNet":
        model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
        model = nn.DataParallel(model)
        checkpoint = torch.load(trained_model_path)
        model.load_state_dict(checkpoint['model_state_dict'], False)
        model = model.to(device)
        loaded_models = [model]
    else:
        print('model error')
    print("Loaded pretrained model {}".format(which_model))

  for data in data_dirs:
    count += 1
    if count < 0:
      continue
    print('---------')
    print(data)
    radar_idx = file_to_id(data)

    raw_radars = list()
    gt_data_handle = np.load(data, allow_pickle=True)
    gt_dict = gt_data_handle.item()

    num_past_pcs = num_past_frames
    for i in range(num_past_pcs):
      radar_img = gt_dict['raw_radar_' + str(i)]
      if motion_comp:
        if i > 0:
          tf = np.array(gt_dict['gt_tf_' + str(i)]) # w.r.t. disp frame
          cv_x = tf[0]/cart_resolution
          cv_y = tf[1]/cart_resolution
          cv_theta = -tf[2]
          affine_mat = np.zeros((2,3))
          affine_mat[0] = [np.cos(cv_theta), np.sin(cv_theta), cv_x*np.cos(cv_theta) + cv_y*np.sin(cv_theta) + ( (1-np.cos(cv_theta))*radar_img.shape[1]/2 - np.sin(cv_theta)*radar_img.shape[0]/2 )]
          affine_mat[1] = [-np.sin(cv_theta), np.cos(cv_theta), cv_x*(-np.sin(cv_theta)) + cv_y*np.cos(cv_theta) + ( np.sin(cv_theta)*radar_img.shape[1]/2 + (1-np.cos(cv_theta))*radar_img.shape[0]/2 )]
          dst = cv2.warpAffine(radar_img, affine_mat, radar_img.shape)
          radar_img = dst
          #fig, ax = plt.subplots(1, 1, figsize=(6, 6))
          #ax.imshow(radar_img)
          #plt.show()
      raw_radars.append(np.expand_dims(radar_img, axis=2))

    raw_radars = np.stack(raw_radars, 0).astype(np.float32)
    raw_radars_list = list()
    raw_radars_list.append(raw_radars)
    raw_radars = np.stack(raw_radars_list, 0)
    raw_radars = torch.tensor(raw_radars)

    raw_radars_curr = torch.from_numpy(np.zeros((raw_radars.shape[0], 1, 256, 256, 1)).astype(np.float32))
    raw_radars_curr[:,0,:,:,:] = raw_radars[:,0,:,:,:]
    raw_radars_curr = raw_radars_curr.to(device)

    raw_radar = raw_radars[0,0,:,:,0]
    raw_radar_power_mask = np.copy(raw_radar)
    raw_radar_power_mask[raw_radar_power_mask>0.08]=1 #0.08
    raw_radar_power_mask[raw_radar_power_mask<=0.08]=0
    radar_gt = gt_dict['gt_radar']
    radar_gt[radar_gt > 0] = 1
    motion_gt = gt_dict['gt_moving']
    motion_gt[motion_gt > 0] = 1
    disp_gt = gt_dict['gt_disp_0']
    lidar_gt = gt_dict['gt_lidar'].astype(np.bool)
    disp_gt_global = gt_dict['gt_disp_global']*4 * np.expand_dims(motion_gt,axis=2)

    ## gt thres by power mask
    radar_gt_power_mask = radar_gt*raw_radar_power_mask
    motion_gt_power_mask = motion_gt*raw_radar_power_mask

    radar_gt_ = torch.from_numpy(radar_gt).to(device).type(torch.FloatTensor)
    radar_gt_power_mask_ = torch.from_numpy(radar_gt_power_mask).to(device).type(torch.FloatTensor)
    motion_gt_power_mask_ = torch.from_numpy(motion_gt_power_mask).to(device).type(torch.FloatTensor)*radar_gt_power_mask_
    ######disp_gt_ = torch.from_numpy(disp_gt).to(device).type(torch.FloatTensor)*torch.unsqueeze(radar_gt_power_mask_,2)

    ## calc_ego_motion_disp_map ##
#    ego_motion_disp_map = calc_ego_motion_disp_map(gt_dict, num_past_pcs)
    disp_gt_global_ = torch.from_numpy(disp_gt-ego_motion_disp_map).to(device).type(torch.FloatTensor)*torch.unsqueeze(radar_gt_power_mask_,2)
#    print(disp_gt_global_.shape)
    #disp_gt_global_=-1
    ######disp_gt_global_ = torch.from_numpy(disp_gt_global).to(device).type(torch.FloatTensor)*torch.unsqueeze(radar_gt_power_mask_,2)
#    f, a = plt.subplots(1, 2, figsize=(10, 5))
#    a[0].imshow(disp_gt_global_[...,1].cpu().numpy())
#    a[1].imshow(gt_dict['gt_disp_global'][...,1])
#    #a[1].imshow(gt_dict['gt_disp_global'][...,1].cpu().numpy())
#    plt.show()

    if not opt:
      ### network prediction ###
      model = loaded_models[0]
      model.eval()
      with torch.no_grad():
        # network estimate #
        if use_temporal_info:
          disp_pred, cat_pred, motion_pred = model(raw_radars)
        else:
          disp_pred, cat_pred, motion_pred = model(raw_radars_curr)

      cat_pred_ = torch.logical_not(torch.argmax(cat_pred[0], axis=0)).type(torch.FloatTensor)
      motion_pred_ = torch.logical_not(torch.argmax(motion_pred[0], axis=0)).type(torch.FloatTensor)
      motion_pred_ = motion_pred_ * radar_gt_ # *radar_gt_ # * radar_gt_power_mask_
      disp_pred_ = disp_pred[0].type(torch.FloatTensor).permute(1,2,0)*torch.unsqueeze(radar_gt_power_mask_,2)
    else:
      ### opticalflow prediction ###
      disp_pred_opt, motion_pred_opt = opticalflow_pred(raw_radars, radar_gt, 5.)
      #disp_pred_opt = np.zeros((256,256,2))
      #motion_pred_opt = np.zeros((256,256))
      #data = tmp_path+'/'+str(radar_idx)+'.npy'
      #res_data_handle = np.load(data, allow_pickle=True)
      #res_dict = res_data_handle.item()
      #motion_res = res_dict['gt_moving']
      #motion_res[motion_res > 0] = 1
      #motion_pred_opt = motion_res
      motion_pred_opt_ = torch.from_numpy(motion_pred_opt).type(torch.FloatTensor)*radar_gt_power_mask_
      disp_pred_opt_ = torch.from_numpy(disp_pred_opt).type(torch.FloatTensor)*torch.unsqueeze(radar_gt_power_mask_,2)

    ### error calculation ### >> use power mask
#    if not opt:
#      calc_err(radar_gt_power_mask_, motion_gt_power_mask_, disp_gt_, disp_gt_global_, cat_pred_, motion_pred_, disp_pred_, device)
#    else:
#      calc_err(radar_gt_power_mask_, motion_gt_power_mask_, disp_gt_, disp_gt_global_, radar_gt_power_mask_, motion_pred_opt_, disp_pred_opt_, device)

    if save or disp:
      # convert all output to numpy
      if not opt:
        cat_pred_numpy = cat_pred_.cpu().numpy()
        motion_pred_numpy = motion_pred_.cpu().numpy()
        disp_pred_numpy = disp_pred_.cpu().numpy()
      else:
        cat_pred_numpy = radar_gt
        motion_pred_numpy = motion_pred_opt
        disp_pred_numpy = disp_pred_opt

      raw_radars = raw_radars.detach().numpy()
      raw_radar = raw_radar.detach().numpy()

      # compute odometry by disp pred
      #M, outlier_mask = calc_odom_by_disp_map(disp_pred_numpy[...,0], disp_pred_numpy[...,1], cat_pred_numpy[0,0,:,:], 3.0)

      # visualize network output #
      viz_cat_pred = cat_pred_numpy

      viz_motion_pred = motion_pred_numpy
      #viz_motion_pred = outlier_mask

      if save:
        fig1, ax1 = plt.subplots(1, 1, figsize=(14, 14))
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 14))
        fig3, ax3 = plt.subplots(1, 1, figsize=(14, 14))

      t1 = time.time()

      if plot_class:
        ## plot occupied map
        ax1.imshow(viz_err_fig(raw_radar, radar_gt, viz_cat_pred))
        ax1.axis('off')
        ax1.set_aspect('equal')
        ax1.title.set_text('occupied')

        if save and not opt:
          fig1.savefig(os.path.join(img_save_dir+'/occupied', str(radar_idx) + '.png'), bbox_inches='tight')
          plt.close(fig1)

      if plot_motion:
        ## plot motion segmentation
        # viz_motion_pred*radar_gt
        #ax2.imshow(viz_err_fig(raw_radar, motion_gt, viz_motion_pred))
        ax2.imshow(viz_err_fig(raw_radar, motion_gt_power_mask, viz_motion_pred*radar_gt_power_mask))
        ax2.axis('off')
        ax2.set_aspect('equal')
        ax2.title.set_text('motion segmentation')
        if save:
          fig2.savefig(os.path.join(img_save_dir+'/motion_seg', str(radar_idx) + '.png'), bbox_inches='tight')
          plt.close(fig2)

      if plot_flow:
        ## plot disp map
        raw_radar_viz_img = np.zeros((256,256,3))
        raw_radar_viz_img = np.stack((raw_radar,raw_radar,raw_radar), axis=2)
        #ax3.imshow(raw_radar_viz_img*np.expand_dims(radar_gt,axis=2)*2.)
        ax3.imshow(raw_radar_viz_img*2.)
        ax3.axis('off')
        ax3.set_aspect('equal')
        ax3.title.set_text('flow')
        # compensate disp map #
#        disp_gt = disp_gt - ego_motion_disp_map
#        disp_pred_numpy = disp_pred_numpy - ego_motion_disp_map

        # compute consistency map between gt and res #
        #err_map = np.sqrt(np.sum((disp_gt-disp_pred_numpy)**2,axis=2))
        err_map = np.sqrt(np.sum((disp_gt_global-disp_pred_numpy)**2,axis=2))

        consistency_mask = err_map<=2 #1
        consistency_map = np.zeros((256,256))
        consistency_map[consistency_mask]=1
        #consistency_map = np.zeros((256,256))

        #consistency_qk1 = plot_quiver(ax3, -disp_gt[:,:,0], -disp_gt[:,:,1], radar_gt*consistency_map, color='lime')
        #gt_qk1 = plot_quiver(ax3, -disp_gt[:,:,0], -disp_gt[:,:,1], radar_gt*np.logical_not(consistency_map), color='r')
        #qk1 = plot_quiver(ax3, -disp_pred_numpy[...,0], -disp_pred_numpy[...,1], radar_gt*np.logical_not(consistency_map), color='deepskyblue') # deepskyblue

        consistency_qk1 = plot_quiver(ax3, disp_gt_global[:,:,0], disp_gt_global[:,:,1], radar_gt*consistency_map, color='lime')
        gt_qk1 = plot_quiver(ax3, disp_gt_global[:,:,0], disp_gt_global[:,:,1], radar_gt*np.logical_not(consistency_map), color='r')
        qk1 = plot_quiver(ax3, disp_pred_numpy[...,0], disp_pred_numpy[...,1], radar_gt*np.logical_not(consistency_map), color='deepskyblue') # deepskyblue

        ## plot flow
  #      opticalflow_img = flow_to_img(flow[...,0], flow[...,1])
  #      opticalflow_viz = (opticalflow_img+raw_radar_viz_img)/1.
  #      opticalflow_viz[opticalflow_viz>1.] = 1.
  #      cv2.namedWindow('opticalflow_img', cv2.WINDOW_NORMAL)
  #      cv2.imshow('opticalflow_img', (opticalflow_img+raw_radar_viz_img)/1.)
  #      disp_pred_img = flow_to_img(disp_pred_numpy[0,0,...], -disp_pred_numpy[0,1,...])
  #      disp_pred_img = disp_pred_img * np.expand_dims(radar_gt, axis=2)
  #      disp_pred_viz = (disp_pred_img+raw_radar_viz_img)/1
  #      disp_pred_viz[disp_pred_viz>1.] = 1.
  #      cv2.namedWindow('disp_pred_img', cv2.WINDOW_NORMAL)
  #      cv2.imshow('disp_pred_img', (disp_pred_img+raw_radar_viz_img)/1.)
        if save:
          fig3.savefig(os.path.join(img_save_dir+'/flow', str(radar_idx) + '.png'), bbox_inches='tight')
          plt.close(fig3)

      t2 = time.time()
      print('plotting time cost:',t2-t1)

    if disp:
      cv2.waitKey(50)
      plt.pause(0.1)
    if save:
      plt.close()

    if disp:
      ax1.clear
      ax2.clear
      ax3.clear
      consistency_qk1.remove()
      gt_qk1.remove()
      qk1.remove()
    frame_idx = frame_idx + 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', default=None, type=str, help='The path to nuScenes dataset')
    parser.add_argument('-v', '--version', default='v1.0-trainval', type=str, help='The version of nuScenes dataset')
    #parser.add_argument('-l', '--savepath', default=None, type=str, help='Directory for saving the generated images')
    parser.add_argument('-n', '--nframe', default=10, type=int, help='The number of frames to be generated')
    parser.add_argument('-s', '--scene', default=5, type=int, help='Which scene')
    parser.add_argument('--net', default='RaMNet', type=str, help='Which network [MotionNet/MotionNetMGDA]')
    parser.add_argument('--modelpath', default=None, type=str, help='Path to the pretrained model')
    parser.add_argument('--beginframe', default=0, type=int, help='From which frame we start predicting')
    parser.add_argument('--format', default='gif', type=str, help='The output animation format [gif/mp4]')

    parser.add_argument('--video', action='store_true', help='Whether to generate images or [gif/mp4]')
#    parser.add_argument('--adj', action='store_false', help='Whether predict the relative offset between frames')
    parser.add_argument('--disp', action='store_true', help='Whether to immediately show the images')
    parser.add_argument('--save', action='store_true', help='Whether to save the images')
#    parser.add_argument('--jitter', action='store_false', help='Whether to apply jitter suppression')
    parser.add_argument('--opt', action='store_true', help='Run optical flow')
    args = parser.parse_args()

    gen_prediction_frames = not args.video
    if_disp = args.disp
    if_save = args.save
    if_opt = args.opt
    #image_save_dir = check_folder(args.savepath)

    class_err_avg = AverageMeter('class_err', ':.7f')
    motion_err_avg = AverageMeter('motion_err', ':.7f')
    disp_err_avg = AverageMeter('disp_err', ':.7f')
    moving_disp_err_avg = AverageMeter('moving_disp_err', ':.7f')
    static_disp_err_avg = AverageMeter('static_disp_err', ':.7f')

    motion_speed0_err_avg = AverageMeter('motion_speed0_err', ':.7f')
    motion_speed1_err_avg = AverageMeter('motion_speed1_err', ':.7f')
    motion_speed2_err_avg = AverageMeter('motion_speed2_err', ':.7f')

    disp0_err_avg = AverageMeter('disp0_err', ':.7f')
    disp1_err_avg = AverageMeter('disp1_err', ':.7f')
    disp2_err_avg = AverageMeter('disp2_err', ':.7f')

    if if_save:
      matplotlib.use('Agg')
    if if_disp:
      #fig, ax = plt.subplots(1, 2, figsize=(17, 11)) # 20 7
      fig1, ax1 = plt.subplots(1, 1, figsize=(7, 7))
      fig2, ax2 = plt.subplots(1, 1, figsize=(7, 7))
      fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))
      plt.tight_layout()
    else:
      ax1=-1
      ax2=-1
      ax3=-1

    if gen_prediction_frames:
        vis_result(data_path=args.data, trained_model_path=args.modelpath, which_model=args.net, disp=if_disp, opt=if_opt, save=if_save, ax1=ax1, ax2=ax2, ax3=ax3)
    else:
        frames_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        save_gif_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format='gif')
