import os
import numpy as np
import matplotlib.pyplot as plt

# np.rot90 => counterclockwise

#dataset_root = '/mnt/Disk1/training_data_local_global_vel_i5o4_with_tf_no_mask/2019-01-10-11-46-21-radar-oxford-10k'
#save_directory = '/mnt/Disk1/training_data_local_global_vel_i5o4_with_tf_no_mask/data_aug/2019-01-10-11-46-21-radar-oxford-10k'

dataset_root = '/home/950154_customer/frank/data/itri_global_vel_i2o1_no_mask/2021-05-05-18-24-24'
save_directory = '/home/950154_customer/frank/data/itri_global_vel_i2o1_no_mask/data_aug/2021-05-05-18-24-24'


print("data root:", dataset_root)

dims = np.array([256, 256])
num_past_frames = 2 #2
num_future_frames = 1 #1

seq_dirs = [os.path.join(dataset_root, f)for f in os.listdir(dataset_root)
                 if os.path.isfile(os.path.join(dataset_root, f))]

print(len(seq_dirs))

for gt_file_path in seq_dirs:
  print(gt_file_path)
  idx = gt_file_path.split("/")[-1].split(".")[0]
  #print(idx, f, float(idx+'.'+f))
#  if float(idx)<400:
#    contiue

#f = gt_file_path.split("/")[-1].split(".")[1]
#  if float(idx+'.'+f)<1620211822.381726632:
#    continue
  gt_data_handle = np.load(gt_file_path, allow_pickle=True)
  gt_dict = gt_data_handle.item()

  for num_rot in range(4):
    #num_rot = 0
    save_data_dict = dict()
    save_data_dict['gt_radar'] = np.rot90(gt_dict['gt_radar'], num_rot)
    save_data_dict['gt_moving'] = np.rot90(gt_dict['gt_moving'], num_rot)
    save_data_dict['gt_lidar'] = np.rot90(gt_dict['gt_lidar'], num_rot)
    for i in range(num_past_frames):
      save_data_dict['raw_radar_' + str(i)] = np.rot90(gt_dict['raw_radar_' + str(i)], num_rot)
#    for i in range(num_future_frames):
#      pixel_disp_map = gt_dict['gt_disp_'+str(i)]
#      pixel_disp_map_rot = np.rot90(pixel_disp_map, num_rot)
#      pixel_disp_map_new = np.zeros_like(pixel_disp_map)
#      if num_rot == 0:
#        pixel_disp_map_new = pixel_disp_map_rot
#      if num_rot == 1:
#        pixel_disp_map_new[:,:,0] = -pixel_disp_map_rot[:,:,1]
#        pixel_disp_map_new[:,:,1] = pixel_disp_map_rot[:,:,0]
#      if num_rot == 2:
#        pixel_disp_map_new[:,:,0] = -pixel_disp_map_rot[:,:,0]
#        pixel_disp_map_new[:,:,1] = -pixel_disp_map_rot[:,:,1]
#      if num_rot == 3:
#        pixel_disp_map_new[:,:,0] = pixel_disp_map_rot[:,:,1]
#        pixel_disp_map_new[:,:,1] = -pixel_disp_map_rot[:,:,0]
#      save_data_dict['gt_disp_' + str(i)] = pixel_disp_map_new

    pixel_disp_global_map = gt_dict['gt_disp_global']
    pixel_disp_global_map_rot = np.rot90(pixel_disp_global_map, num_rot)
    pixel_disp_global_map_new = np.zeros_like(pixel_disp_global_map)
    if num_rot == 0:
      pixel_disp_global_map_new = pixel_disp_global_map_rot
    if num_rot == 1:
      pixel_disp_global_map_new[:,:,0] = -pixel_disp_global_map_rot[:,:,1]
      pixel_disp_global_map_new[:,:,1] = pixel_disp_global_map_rot[:,:,0]
    if num_rot == 2:
      pixel_disp_global_map_new[:,:,0] = -pixel_disp_global_map_rot[:,:,0]
      pixel_disp_global_map_new[:,:,1] = -pixel_disp_global_map_rot[:,:,1]
    if num_rot == 3:
      pixel_disp_global_map_new[:,:,0] = pixel_disp_global_map_rot[:,:,1]
      pixel_disp_global_map_new[:,:,1] = -pixel_disp_global_map_rot[:,:,0]
    save_data_dict['gt_disp_global'] = pixel_disp_global_map_new

#    for i in range(num_past_frames):
#      tf = gt_dict['gt_tf_'+str(i)]
#      tf_new = [0,0,0]
#      if num_rot == 0:
#        tf_new = tf
#      if num_rot == 1:
#        tf_new[0] = -tf[1]
#        tf_new[1] = tf[0]
#        tf_new[2] = tf[2]
#      if num_rot == 2:
#        tf_new[0] = -tf[0]
#        tf_new[1] = -tf[1]
#        tf_new[2] = tf[2]
#      if num_rot == 3:
#        tf_new[0] = tf[1]
#        tf_new[1] = -tf[0]
#        tf_new[2] = tf[2]
#      save_data_dict['gt_tf_' + str(i)] = tf_new
#      print(tf)
#      print(tf_new)

    save_file_name = os.path.join(save_directory, idx+'_'+str(num_rot) + '.npy')
    np.save(save_file_name, arr=save_data_dict)

    #disp_glo = save_data_dict['gt_disp_global'] * np.expand_dims(save_data_dict['gt_radar'], axis=2)
    #disp_glo_norm = np.linalg.norm(disp_glo, ord=2, axis=-1)
    #fig, ax = plt.subplots(1, 4, figsize=(10, 14))
    #ax[0].imshow(save_data_dict['raw_radar_0'])
    #ax[1].imshow(save_data_dict['gt_radar'])
    #ax[2].imshow(save_data_dict['gt_moving'])
    #ax[1].imshow(disp_glo_norm)
    #ax[2].imshow(disp_glo[:,:,0])
    #ax[3].imshow(disp_glo[:,:,1])
    #plt.show()

