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

plot_class = 1
plot_motion = 0
plot_flow = 0

motion_comp = False

img_save_dir = '/mnt/Disk1/viz_result_3fig/2021-05-18_23-04-38_ep10_test_polar'
num_past_frames = 1
out_seq_len = 1
height_feat_size=1
plot_motion_seg = False
use_temporal_info = False

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
  viz_err = np.zeros((raw_radar.shape[0],raw_radar.shape[1],3))
  viz_err[:,:,0] = gt * np.logical_not(res)
  viz_err[:,:,1] = gt * res
  viz_err[:,:,2] = res * np.logical_not(gt) *2
  viz_radar = np.stack((raw_radar,raw_radar,raw_radar), axis=2)
  return (viz_err + viz_radar*3.)/2. # *3

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



def find(s, ch):
    return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.jpg')]
  idx = float(idx)
  return idx

def calc_err(result, gt, device):
  result = torch.from_numpy(result).to(device).type(torch.FloatTensor)
  gt = torch.from_numpy(gt).to(device).type(torch.FloatTensor)
  # compute class error #
  class_err = torch.sum((result-gt)**2) / (result.shape[0]*result.shape[1])
  class_err_avg.update(class_err.item())
  print(class_err_avg)
#  f, a = plt.subplots(1, 2, figsize=(10, 5))
#  a[0].imshow(result.cpu().numpy())
#  a[1].imshow(gt.cpu().numpy())
#  plt.show()



def img_polar_to_cart(img):
  img = np.concatenate((img[int(img.shape[0]/4):,:], img[:int(img.shape[0]/4),:]), axis=0)
  value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
  cart_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
  cart_image = cv2.resize(cart_image, (256, 256), interpolation=cv2.INTER_LINEAR)
  return cart_image
def img_cart_to_polar_extend(img):
  img = cv2.resize(img, (260, 260), interpolation=cv2.INTER_LINEAR)
  img = np.rot90(img,-1) # rot clockwise 90 deg
  value = np.sqrt(((img.shape[0]/2.0)**2.0)+((img.shape[1]/2.0)**2.0))
  polar_image = cv2.linearPolar(img,(img.shape[0]/2, img.shape[1]/2), value, cv2.WARP_FILL_OUTLIERS)
  return polar_image

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

def polar_to_cart_test(fft_data, cart_resolution, cart_pixel_width, range_resolution, angle_resolution):
  interpolate_crossover  = True

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
  azimuths_0 = angle_resolution # 0.02416609734 #
  azimuth_step = angle_resolution # 0.02416609734 # 2*pi/260
  sample_u = (sample_range - range_resolution / 2) / range_resolution
  sample_v = (sample_angle - azimuths_0) / azimuth_step

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


def network_pred(radar_img, model, mode='cat'):
  ### input to network
  raw_radars = np.expand_dims(radar_img, axis=(0,4))
  raw_radars = torch.tensor(raw_radars)
  ### network prediction ###
  model.eval()
  with torch.no_grad():
    disp_pred, cat_pred, motion_pred = model(raw_radars)
  if mode == 'cat':
    cat_pred_ = torch.logical_not(torch.argmax(cat_pred[0], axis=0)).type(torch.FloatTensor)
    cat_pred_numpy = cat_pred_.cpu().numpy()
    return cat_pred_, cat_pred_numpy
  elif mode == 'motion':
    mo_pred_ = torch.logical_not(torch.argmax(motion_pred[0], axis=0)).type(torch.FloatTensor)
    mo_pred_numpy = mo_pred_.cpu().numpy()
    return mo_pred_, mo_pred_numpy

def estimate_full_scan_w_angle_normalized(radar_img, model, sec):
  print(radar_img.shape)
  radar_img_result = np.zeros_like(radar_img)
  radar_img_result_viz = np.zeros_like(radar_img)
  for i in range(sec):
    #print('n:', i+1)
    radar_img_range_sec = radar_img[:,256*i:256*(i+1)]
    angle_sec_num = 2**i
    radar_img_range_sec_result = np.zeros_like(radar_img_range_sec)
    radar_img_range_sec_result_viz = np.zeros_like(radar_img_range_sec)
    for j in range(angle_sec_num):
      #print('j:', j+1)
      angle_res = int(256/angle_sec_num)
      radar_img_range_angle_sec = radar_img_range_sec[angle_res*j:angle_res*(j+1),:]
      #print(radar_img_range_angle_sec.shape)
      radar_img_range_angle_sec_input = cv2.resize(radar_img_range_angle_sec, (256,256))
      radar_img_range_angle_sec_output_torch, radar_img_range_angle_sec_output = network_pred(radar_img_range_angle_sec_input, model)
      radar_img_range_angle_sec_result = cv2.resize(radar_img_range_angle_sec_output, (radar_img_range_angle_sec.shape[1],radar_img_range_angle_sec.shape[0]))
      #print(radar_img_range_angle_sec_result.shape)
      radar_img_range_angle_sec_result_viz = np.copy(radar_img_range_angle_sec_result)
      # Plot boundary
      #radar_img_range_angle_sec_result[0,:]=1
      width = 1
      radar_img_range_angle_sec_result_viz[:,0:width]=1
      radar_img_range_angle_sec_result_viz[:,-1-width:-1]=1

      radar_img_range_sec_result[angle_res*j:angle_res*(j+1),:] = radar_img_range_angle_sec_result
      radar_img_range_sec_result_viz[angle_res*j:angle_res*(j+1),:] = radar_img_range_angle_sec_result_viz
    radar_img_result[:,256*i:256*(i+1)] = radar_img_range_sec_result
    radar_img_result_viz[:,256*i:256*(i+1)] = radar_img_range_sec_result_viz
  return radar_img_result, radar_img_result_viz

def estimate_full_scan(radar_img, model, sec):
  radar_img_result = np.zeros_like(radar_img)
  radar_img_result_viz = np.zeros_like(radar_img)
  for i in range(5):
    start_dis = 32*i
    gt_img_power_mask = np.zeros_like(radar_img) # useless
    radar_img_result_range_sec, radar_img_range_sec, gt_img_power_mask_range_sec = estimate_sliding_window_scan(radar_img, gt_img_power_mask, model, start_dis)
    radar_img_result_range_sec_viz = np.copy(radar_img_result_range_sec)
    width=2
    radar_img_result_range_sec_viz[:,0:width]=1
    radar_img_result_range_sec_viz[:,-1-width:-1]=1
    radar_img_result[:,256*i:256*(i+1)] = radar_img_result_range_sec
    radar_img_result_viz[:,256*i:256*(i+1)] = radar_img_result_range_sec_viz
  return radar_img_result, radar_img_result_viz

def estimate_sliding_window_scan_w_angle_normalized(radar_img, gt_img_power_mask, model, start_dis):
  n = round(start_dis/0.125)
  radar_img_range_sec = radar_img[:,n:n+256]
  gt_img_range_sec_power_mask = gt_img_power_mask[:,n:n+256]

  angle_sec_num = 2**int(start_dis/32)
  radar_img_range_sec_result = np.zeros_like(radar_img_range_sec)
  for j in range(angle_sec_num):
    #print('j:', j+1)
    angle_res = int(256/angle_sec_num)
    radar_img_range_angle_sec = radar_img_range_sec[angle_res*j:angle_res*(j+1),:]
    #print(radar_img_range_angle_sec.shape)
    radar_img_range_angle_sec_input = cv2.resize(radar_img_range_angle_sec, (256,256))
    radar_img_range_angle_sec_output_torch, radar_img_range_angle_sec_output = network_pred(radar_img_range_angle_sec_input, model)
    radar_img_range_angle_sec_result = cv2.resize(radar_img_range_angle_sec_output, (radar_img_range_angle_sec.shape[1],radar_img_range_angle_sec.shape[0]))
    #print(radar_img_range_angle_sec_result.shape)
    # Plot boundary
    '''
    radar_img_range_angle_sec_result[0,:]=1
    radar_img_range_angle_sec_result[:,0:3]=1
    radar_img_range_angle_sec_result[:,-4:-1]=1
    '''
    radar_img_range_sec_result[angle_res*j:angle_res*(j+1),:] = radar_img_range_angle_sec_result
  return radar_img_range_sec_result, radar_img_range_sec, gt_img_range_sec_power_mask

def estimate_sliding_window_scan(radar_img, gt_img_power_mask, model, start_dis):
  n = round(start_dis/0.125)
  radar_img = radar_img[:,n:n+256]
  cat_pred_, radar_img_result = network_pred(radar_img, model)
  gt_img_power_mask = gt_img_power_mask[:,n:n+256]
  return radar_img_result, radar_img, gt_img_power_mask



def estimate_full_scan_cart(radar_cart_img, model, mode='cat'):

  sec = int(radar_cart_img.shape[2]/256)
  radar_cart_img_result = np.zeros_like(radar_cart_img[0])
  radar_cart_img_result_viz = np.zeros_like(radar_cart_img[0])
  for i in range(sec):
    for j in range(sec):
      radar_cart_img_part = radar_cart_img[:, i*256:(i+1)*256, j*256:(j+1)*256]
      cat_pred_, radar_cart_img_part_result = network_pred(radar_cart_img_part, model, mode)
      radar_cart_img_result[i*256:(i+1)*256, j*256:(j+1)*256] = radar_cart_img_part_result
      radar_cart_img_part_result_viz = np.copy(radar_cart_img_part_result)
      radar_cart_img_part_result_viz[0,:]=1
      radar_cart_img_part_result_viz[-1,:]=1
      radar_cart_img_part_result_viz[:,0]=1
      radar_cart_img_part_result_viz[:,-1]=1
      radar_cart_img_result_viz[i*256:(i+1)*256, j*256:(j+1)*256] = np.copy(radar_cart_img_part_result_viz)
  return radar_cart_img_result, radar_cart_img_result_viz


def vis_result(data_path, gt_path, gt_cart32_path, gt_cart64_path, trained_model_path, disp, save, ax1=-1, ax2=-1, ax3=-1, ax4=-1):
  global frame_idx, count

  data_dirs = [os.path.join(data_path, f)for f in os.listdir(data_path)
                   if os.path.isfile(os.path.join(data_path, f))]
  data_dirs.sort(key=file_to_id)
  gt_dirs = [os.path.join(gt_path, f)for f in os.listdir(gt_path)
                   if os.path.isfile(os.path.join(gt_path, f))]
  gt_dirs.sort(key=file_to_id)
  gt_cart32_dirs = [os.path.join(gt_cart32_path, f)for f in os.listdir(gt_cart32_path)
                   if os.path.isfile(os.path.join(gt_cart32_path, f))]
  gt_cart32_dirs.sort(key=file_to_id)
  gt_cart64_dirs = [os.path.join(gt_cart64_path, f)for f in os.listdir(gt_cart64_path)
                   if os.path.isfile(os.path.join(gt_cart64_path, f))]
  gt_cart64_dirs.sort(key=file_to_id)
  print(len(data_dirs), len(gt_dirs))

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=use_temporal_info, num_past_frames=num_past_frames)
  model = nn.DataParallel(model)
  checkpoint = torch.load(trained_model_path)
  model.load_state_dict(checkpoint['model_state_dict'], False)
  model = model.to(device)

  motion_model = RaMNet(out_seq_len=out_seq_len, motion_category_num=2, cell_category_num=2, height_feat_size=height_feat_size, use_temporal_info=True, num_past_frames=2)
  motion_model = nn.DataParallel(motion_model)
  checkpoint = torch.load('/mnt/Disk1/trained_model/train_multi_seq/2021-04-28_20-10-39/epoch_10.pth')
  motion_model.load_state_dict(checkpoint['model_state_dict'], False)
  motion_model = motion_model.to(device)

  for count in range(len(data_dirs)):
    count += 1
    if count < 0 or count%2==0 or count > 4000: # only use half dataset
      continue
    print('---------')

    data_path = data_dirs[count]
    data_path_old = data_dirs[count-1]
    gt_path = gt_dirs[count]
    gt_cart32_path = gt_cart32_dirs[count]
    gt_cart64_path = gt_cart64_dirs[count]
    print(data_path, '\n', gt_path)

    radar_img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    radar_img_old = cv2.imread(data_path_old, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    gt_cart32_img = cv2.imread(gt_cart32_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    gt_cart64_img = cv2.imread(gt_cart64_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.
    #gt_cart_img = gt_cart64_img[:256,128:128*3] ### front 32m
    gt_cart_img = gt_cart64_img[:256,128:128*3] ### front 32m


    ##### processing data #####
    radar_img_raw = radar_img[:,11:] # rm metadata
    radar_cart32_img = polar_to_cart_test(radar_img_raw, 0.25, 256, 0.0432, 2*math.pi/400).squeeze()
    radar_cart64_img = polar_to_cart_test(radar_img_raw, 0.25, 512, 0.0432, 2*math.pi/400).squeeze()
    radar_cart96_img = polar_to_cart_test(radar_img_raw, 0.25, 768, 0.0432, 2*math.pi/400).squeeze()
    radar_cart_full_img = polar_to_cart_test(radar_img_raw, 0.25, 1280, 0.0432, 2*math.pi/400).squeeze()
    #radar_cart_img = radar_cart64_img[:256,128:128*3] ### front 32m
    #radar_cart_img = radar_cart96_img[:256,256:256*2] ### front far 32m
    radar_cart_img = radar_cart_full_img
    radar_cart_img_ = np.expand_dims(radar_cart_img, axis=0)

    radar_img_old_raw = radar_img_old[:,11:] # rm metadata
    radar_cart_full_img_old = polar_to_cart_test(radar_img_old_raw, 0.25, 1280, 0.0432, 2*math.pi/400).squeeze()
    radar_cart_img_multi_ = np.concatenate((np.expand_dims(radar_cart_full_img, axis=0), np.expand_dims(radar_cart_full_img_old, axis=0)), axis=0)

#    radar_power_mask = np.copy(radar_cart_img)
#    radar_power_mask[radar_power_mask>0.08]=1 #0.08
#    radar_power_mask[radar_power_mask<=0.08]=0
#    gt_cart_img_power_mask = gt_cart_img * radar_power_mask
    ##### end processing data #####

    # full cart test
    radar_cart_img_result, radar_cart_img_result_viz = estimate_full_scan_cart(radar_cart_img_, model)

    motion_cart_img_result, motion_cart_img_result_viz = estimate_full_scan_cart(radar_cart_img_multi_, motion_model, mode='motion')
    motion_cart_img_result = motion_cart_img_result * radar_cart_img_result

    if save or disp:
      # visualize network output #
#      if save:
#        fig1, ax1 = plt.subplots(1, 2, figsize=(28, 56))
#        fig2, ax2 = plt.subplots(1, 2, figsize=(28, 56))
#        fig3, ax3 = plt.subplots(1, 1, figsize=(28, 28))
#        fig4, ax4 = plt.subplots(1, 1, figsize=(28, 28))

      t1 = time.time()

      plot_res_cart = viz_err_fig(radar_cart_img, radar_cart_img_result, radar_cart_img_result)[:,:,::-1] #rgb2bgr
      plot_motion_res_cart = viz_err_fig(radar_cart_img, radar_cart_img_result, motion_cart_img_result)[:,:,::-1] #rgb2bgr

      cv2.namedWindow('viz',cv2.WINDOW_NORMAL)
      cv2.imshow('viz', plot_res_cart)
      cv2.namedWindow('viz_motion',cv2.WINDOW_NORMAL)
      cv2.imshow('viz_motion', plot_motion_res_cart)
      cv2.waitKey(0)

#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-24_22-43-05_ep10/full/cart_radar_full/' + str(count) + '.png'), radar_img_cart*2.*255.)
#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-24_22-43-05_ep10/full/cart_res_full/' + str(count) + '.png'), radar_img_result_cart_viz*255.)
#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-24_22-43-05_ep10/full/cart_full/' + str(count) + '.png'), plot_res_cart*255.)
#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-24_22-43-05_ep10/full/polar_full/' + str(count) + '.png'), plot_res_polar*255.)

#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-26_14-30-37_ep15/full/cart_full/' + str(count) + '.png'), plot_res_cart*255.)
#      cv2.imwrite(os.path.join('/mnt/Disk1/viz_polar/2021-05-26_14-30-37_ep15/full/cart_res_full/' + str(count) + '.png'), radar_cart_img_result_viz*255.)


      t2 = time.time()
      print('plotting time cost:',t2-t1)

    if disp:
      cv2.waitKey(50)
      plt.pause(0.1)
    if save:
      plt.close()

    if disp:
      ax1[0].clear
      ax1[1].clear
      ax2[0].clear
      ax2[1].clear
      ax3.clear
      ax4.clear
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
    parser.add_argument('--disp', action='store_true', help='Whether to immediately show the images')
    parser.add_argument('--save', action='store_true', help='Whether to save the images')
    args = parser.parse_args()

    gen_prediction_frames = not args.video
    if_disp = args.disp
    if_save = args.save

    class_err_avg = AverageMeter('class_err', ':.7f')

    weight_avg = AverageMeter('weight', ':.7f')

    if if_save:
      matplotlib.use('Agg')
    if if_disp:
      fig1, ax1 = plt.subplots(1, 2, figsize=(28, 56))
      fig2, ax2 = plt.subplots(1, 2, figsize=(28, 56))
      fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))
      fig4, ax4 = plt.subplots(1, 1, figsize=(7, 7))
      plt.tight_layout()
    else:
      ax1=-1
      ax2=-1
      ax3=-1
      ax4=-1

    if gen_prediction_frames:
#        cv2.namedWindow('viz',cv2.WINDOW_NORMAL)
#        cv2.namedWindow('viz2',cv2.WINDOW_NORMAL)
        dataset = '2019-01-10-12-32-52' # '2019-01-10-11-46-21'
        vis_result(data_path='/mnt/Disk1/dataset/'+dataset+'/radar_raw',
                    gt_path='/mnt/Disk1/dataset/'+dataset+'/lidar_polar',
                    gt_cart32_path='/mnt/Disk1/dataset/'+dataset+'/lidar_cart',
                    gt_cart64_path='/mnt/Disk1/dataset/'+dataset+'/lidar_cart_64',
                    trained_model_path=args.modelpath, disp=if_disp, save=if_save, ax1=ax1, ax2=ax2, ax3=ax3, ax4=ax4)
    else:
        frames_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        save_gif_dir = os.path.join('/media/pwu/Data/3D_data/nuscene/logs/images_job_talk', image_save_dir)
        gen_scene_prediction_video(args.savepath, args.savepath, out_format='gif')
