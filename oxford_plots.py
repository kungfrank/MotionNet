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

import cv2

color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

img_save_dir = '/mnt/Disk1/viz_result/2021-04-30_13-16-22_ep15'
num_past_frames = 2
out_seq_len = 1
height_feat_size=1
plot_motion_seg = True
use_temporal_info = True

global frame_idx
frame_idx = 5

global class_error_sum, motion_error_sum, count
class_error_sum=0.
motion_error_sum=0.
count=0.

def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
  idx = int(idx)
  return idx

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

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



def plot_quiver(ax_, disp0, disp1, viz_cat, viz_motion):
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
  mask = viz_cat.astype(np.bool) * viz_motion.astype(np.bool)
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  qk1 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=15, headaxislength=15, color='r', alpha=0.9, minlength=6, minshaft=1) #'g'

  mask = viz_cat.astype(np.bool) * np.logical_not(viz_motion.astype(np.bool))
  X = idx_x[mask]
  Y = idx_y[mask]
  U = field_gt[:, :, 0][mask]
  V = -field_gt[:, :, 1][mask]
  #qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=10, headaxislength=10, color='mediumblue', alpha=0.9, minlength=6, minshaft=1)
  qk2 = ax_.quiver(Y, X, U, V, angles='xy', scale_units='xy', scale=0.5, width=0.0005, headwidth=30, headlength=10, headaxislength=10, color=[(0.1,0.15,1.0)], alpha=0.9, minlength=6, minshaft=1)
  # dodgerblue
  return qk1, qk2


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

def calc_odom_by_disp_map(disp0, disp1, radar_mask, moving_mask):
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
  M, mask = cv2.findHomography(pc, pc_, cv2.RANSAC, 3.0) # 1~10 => strict~loose 2

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

def vis_result(data_path, trained_model_path, which_model, disp, save, ax=-1):
  global frame_idx
  global class_error_sum, motion_error_sum, count
  data_dirs = [os.path.join(data_path, f)for f in os.listdir(data_path)
                   if os.path.isfile(os.path.join(data_path, f))]
  data_dirs.sort(key=file_to_id)
  print(len(data_dirs))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if count < 0: # 800, 600 -> moving obj # 540 ->left-right moving obj
      continue
    print('---------')
    print(data)
    radar_idx = file_to_id(data)

    raw_radars = list()
    gt_data_handle = np.load(data, allow_pickle=True)
    gt_dict = gt_data_handle.item()

    num_past_pcs = num_past_frames
    for i in range(num_past_pcs):
      raw_radars.append(np.expand_dims(gt_dict['raw_radar_' + str(i)], axis=2))
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

    # gt thres by power mask
    radar_gt = radar_gt*raw_radar_power_mask
    motion_gt = motion_gt*raw_radar_power_mask

    #disp0_gt = gt_dict['gt_disp0']
    #disp1_gt = gt_dict['gt_disp1']
    disp_gt = gt_dict['gt_disp_0']
    lidar_gt = gt_dict['gt_lidar'].astype(np.bool)
    #print(disp_gt.shape)
    #print(lidar_gt.shape)

    model = loaded_models[0]
    model.eval()

    with torch.no_grad():
      # network estimate #
      if use_temporal_info:
        disp_pred, cat_pred, motion_pred = model(raw_radars)
      else:
        disp_pred, cat_pred, motion_pred = model(raw_radars_curr)
    #print('disp_pred:',disp_pred.shape)

#    # compute class error #
#    log_softmax_probs = F.log_softmax(cat_pred, dim=1) # torch.Size([1, 2, 256, 256])
#    pixel_radar_map_gt = gt_to_pixel_map_gt(radar_gt).to(device) # torch.Size([1, 2, 256, 256])
#    loss_class = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
#    loss_class = torch.sum(loss_class) / (cat_pred.shape[2]*cat_pred.shape[3])
#    print('class loss:', loss_class.item())
#    class_error_sum += loss_class.item()
#    print('count:', count, 'avg class loss:', class_error_sum/count)

#    # compute motion error #
#    log_softmax_probs = F.log_softmax(motion_pred, dim=1) # torch.Size([1, 2, 256, 256])
#    pixel_radar_map_gt = gt_to_pixel_map_gt(motion_gt).to(device) # torch.Size([1, 2, 256, 256])
#    loss_motion = torch.sum(- pixel_radar_map_gt * log_softmax_probs, dim=1)
#    loss_motion = torch.sum(loss_motion) / (motion_pred.shape[2]*motion_pred.shape[3])
#    print('motion loss:', loss_motion.item())
#    motion_error_sum += loss_motion.item()
#    print('count:', count, 'avg motion loss:', motion_error_sum/count)

    if save or disp:

      # convert all output to numpy
      cat_pred_numpy = cat_pred.cpu().numpy()
      motion_pred_numpy = motion_pred.cpu().numpy()
      disp_pred_numpy = disp_pred.cpu().numpy()
      raw_radars = raw_radars.detach().numpy()
      raw_radar = raw_radar.detach().numpy()

      # visualize network output #
      viz_cat_pred = cat_pred_numpy[0,:,:,:].argmin(axis=0)

      viz_motion_pred = motion_pred_numpy[0,0,:,:]
      viz_motion_pred[viz_motion_pred>0] = 1
      viz_motion_pred[viz_motion_pred<=0] = 0

      viz_motion_pred = viz_motion_pred * viz_cat_pred
      #viz_motion_pred = viz_motion_pred * radar_gt

      # compute odometry by disp pred
#      M, outlier_mask = calc_odom_by_disp_map(disp_pred_numpy[0,0], disp_pred_numpy[0,1], cat_pred_numpy[0,0,:,:], motion_pred_numpy[0,0,:,:])
#      viz_motion_pred = outlier_mask

      if save:
        fig, ax = plt.subplots(1, 2, figsize=(34, 22)) # 20 7

      t1 = time.time()

      #ax[0].imshow(viz_combined(raw_radar, radar_gt, motion_gt))
      ax[0].imshow(raw_radar, cmap='gray')
      ax[0].axis('off')
      ax[0].set_aspect('equal')
      ax[0].title.set_text('GT')

      ax[1].imshow(viz_combined(raw_radar, viz_cat_pred, viz_motion_pred))
      #ax[1].imshow(viz_combined(raw_radar, radar_gt, motion_gt))
      ax[1].imshow(raw_radar, cmap='gray')
      ax[1].axis('off')
      ax[1].set_aspect('equal')
      ax[1].title.set_text('Result')
      t2 = time.time()
      print('plotting time cost:',t2-t1)
      gt_qk1, gt_qk2 = plot_quiver(ax[0], -disp_gt[:,:,0], -disp_gt[:,:,1], radar_gt, motion_gt)
      qk1, qk2 = plot_quiver(ax[1], -disp_pred_numpy[0,0], -disp_pred_numpy[0,1], viz_cat_pred, viz_motion_pred)

#      zero_mat = np.zeros((256,256))
#      gt_qk1, gt_qk2 = plot_quiver(ax[0], zero_mat, zero_mat, radar_gt, motion_gt)
#      qk1, qk2 = plot_quiver(ax[1], zero_mat, zero_mat, radar_gt, viz_motion_pred)

      raw_radar_viz_img = np.zeros((256,256,3))
      raw_radar_viz_img = np.stack((raw_radar,raw_radar,raw_radar), axis=2)

#      prvs_ = (raw_radars[0,0,:,:,0]*255.).astype(np.uint8)
#      next_ =  (raw_radars[0,1,:,:,0]*255.).astype(np.uint8)
#      flow = cv2.calcOpticalFlowFarneback(prvs_,next_, None, 0.5, 3, 12, 3, 5, 1.2, 0)
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

#      cv2.namedWindow('radar_img', cv2.WINDOW_NORMAL)
#      cv2.imshow('radar_img', raw_radar_viz_img)

    if disp:
      cv2.waitKey(50)
      plt.pause(0.1)

    if save:
      t1 = time.time()
      plt.savefig(os.path.join(img_save_dir, str(radar_idx) + '.png'), bbox_inches='tight')
      #cv2.imwrite(img_save_dir+'/opticalflow/'+str(radar_idx)+'.png', (opticalflow_viz*255).astype(np.uint8))
      #cv2.imwrite(img_save_dir+'/predflow/'+str(radar_idx)+'.png', (disp_pred_viz*255).astype(np.uint8))
      t2 = time.time()
      print('saving time cost:',t2-t1)
      plt.close()
    if save or disp:
      ax[0].clear
      ax[1].clear
      gt_qk1.remove()
      gt_qk2.remove()
      qk1.remove()
      qk2.remove()
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
    args = parser.parse_args()

    gen_prediction_frames = not args.video
    if_disp = args.disp
    if_save = args.save
    #image_save_dir = check_folder(args.savepath)

    if if_save:
      matplotlib.use('Agg')
    if if_disp:
      fig, ax = plt.subplots(1, 2, figsize=(17, 11)) # 20 7
      plt.tight_layout()
    else:
      ax=-1

    if gen_prediction_frames:
        vis_result(data_path=args.data, trained_model_path=args.modelpath, which_model=args.net, disp=if_disp, save=if_save, ax=ax)
    else:
        frames_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        save_gif_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format='gif')
