import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def find(s, ch):
  return [i for i, ltr in enumerate(s) if ltr == ch]
def file_to_id(str_):
  idx = str_[find(str_,'/')[-1]+1:str_.find('.npy')]
  idx = int(idx)
  return idx

logname = '2019-01-10-11-46-21-radar-oxford-10k'
dataset_root1 = '/mnt/Disk1/training_data_mo_seg_0.05_no_mask_labeled/'+logname
dataset_root2 = '/mnt/Disk1/training_data_local_global_vel_i5o4_with_tf_no_mask/'+logname
save_directory = '/mnt/Disk1/training_data_local_global_vel_i5o4_with_tf_no_mask_labeled/'+logname

seq1_dirs = [os.path.join(dataset_root1, f)for f in os.listdir(dataset_root1)
                 if os.path.isfile(os.path.join(dataset_root1, f))]
seq1_dirs.sort(key=file_to_id)

seq2_dirs = [os.path.join(dataset_root2, f)for f in os.listdir(dataset_root2)
                 if os.path.isfile(os.path.join(dataset_root2, f))]
seq2_dirs.sort(key=file_to_id)

print('len(seq_dirs):', len(seq1_dirs), len(seq2_dirs))
#assert len(seq1_dirs) == len(seq2_dirs)

seq_len = len(seq1_dirs)
#seq_len = 2100 # for testing

for file in seq2_dirs:
  radar_idx = file_to_id(file)
  #print(radar_idx)
  file_path1 = dataset_root1+'/'+str(radar_idx)+'.npy'
  file_path2 = dataset_root2+'/'+str(radar_idx)+'.npy'

  if not os.path.exists(file_path1):
    print('File Not Exist. ', file_path1)
    continue
  if not os.path.exists(file_path2):
    print('File Not Exist. ', file_path2)
    continue
  data_handle1 = np.load(file_path1, allow_pickle=True)
  dict1 = data_handle1.item()
  data_handle2 = np.load(file_path2, allow_pickle=True)
  dict2 = data_handle2.item()

  dict_new = dict2.copy()
  dict_new['gt_moving'] = dict1['gt_moving']

  save_file_name = os.path.join(save_directory, str(radar_idx) + '.npy')
  np.save(save_file_name, arr=dict_new)

#  fig, ax = plt.subplots(1, 2, figsize=(10, 10))
#  ax[0].imshow(dict2['gt_moving'])
#  ax[1].imshow(dict_new['gt_moving'])
#  plt.show()

