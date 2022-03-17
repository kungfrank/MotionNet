import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import matplotlib.pyplot as plt
#import cv2

img_save_dir = '/home/joinet/MotionNet/logs/oxford/'
data_save_dir = '/mnt/Disk1/oxford_res_combined_ts_from_pseudo_lidar/2019-01-10-11-46-21-radar-oxford-10k/'

global frame_idx
frame_idx = 45

#plt.figure('test')
fig, ax = plt.subplots(2, 3, figsize=(10, 7))
#fig_, ax_ = plt.subplots(1, 1, figsize=(5, 5))
#plt.tight_layout()

color_map = {0: 'c', 1: 'm', 2: 'k', 3: 'y', 4: 'r'}
cat_names = {0: 'bg', 1: 'bus', 2: 'ped', 3: 'bike', 4: 'other'}

border_meter = 4
border_pixel = border_meter * 4
x_lim = [-(32 - border_meter), (32 - border_meter)]
y_lim = [-(32 - border_meter), (32 - border_meter)]

voxel_size = (0.25, 0.25, 0.4)

def callback(data):
    global frame_idx
    frame_idx = frame_idx + 1
    if frame_idx%10 != 1:
      return
#    cv2.namedWindow('My Image', cv2.WINDOW_NORMAL)
#    img = cv2.imread(img_save_dir+str(frame_idx)+'.png')
#    cv2.imshow('My Image', img)
#    cv2.waitKey(50)

    file_name = format(data.header.stamp.to_sec(), '.12g')
    while len(file_name)<13:
      file_name = file_name+'0'
    if len(file_name)>13:
      file_name = file_name[:12]
    print(file_name)

    p = data_save_dir+str(file_name)+'.npy'
    data = np.load(p , allow_pickle=True)
#    print(data.item()['disp_pred'].shape) # (256, 256, 2)
#    print(data.item()['cat_pred'].shape) # (5, 256, 256)
#    print(data.item()['motion_pred'].shape) # (1, 256, 256)

    disp_pred = data.item()['disp_pred']
    cat_pred = data.item()['cat_pred']
    motion_pred_numpy = data.item()['motion_pred']
    non_empty_map = data.item()['non_empty_map']
    viz_map = data.item()['viz_map']


    cat_pred = np.argmax(cat_pred, axis=0) + 1
    #cat_pred = (cat_pred * non_empty_map * filter_mask).astype(np.int)
    cat_pred = (cat_pred * non_empty_map).astype(np.int)

    idx_x = np.arange(256)
    idx_y = np.arange(256)
    idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    qk = [None] * len(color_map)  # for quiver key
    qk_ = [None] * len(color_map)  # for quiver key

    ax[0,0].imshow(np.rot90(viz_map), cmap = 'jet') # non_empty_map binary
    ax[0,0].axis('off')
    ax[0,0].set_aspect('equal')
    ax[0,0].title.set_text('LIDAR data')

    motion_pred_numpy = np.squeeze(motion_pred_numpy) * non_empty_map

    ax[0,1].imshow(np.rot90(motion_pred_numpy), cmap = 'gray')
    ax[0,1].axis('off')
    ax[0,1].set_aspect('equal')
    ax[0,1].title.set_text('Motion segmentation output')

    vel_map = np.sqrt(np.power(disp_pred[:,:,0],2)+np.power(disp_pred[:,:,1],2)) * non_empty_map
    upper_bound_mask = vel_map >= 20
    vel_map[upper_bound_mask] = 20

    ax[1,1].imshow(np.rot90(vel_map), cmap = 'jet', vmin=0, vmax=10)
    ax[1,1].axis('off')
    ax[1,1].set_aspect('equal')
    ax[1,1].title.set_text('Velocity map output')

    binary_vel_map = np.ones_like(vel_map)
    thd_mask = vel_map < 0.5
    binary_vel_map[thd_mask] = 0
    ax[1,2].imshow(np.rot90(binary_vel_map), cmap = 'gray')
    ax[1,2].axis('off')
    ax[1,2].set_aspect('equal')
    ax[1,2].title.set_text('Motion segmentation + velocity thres')

    for k in range(len(color_map)):
      # ------------------------ Prediction ------------------------
      # Show the prediction results. We show the cells corresponding to the non-empty one-hot gt cells.
      mask_pred = cat_pred == (k + 1)
      field_pred = disp_pred  # Show last prediction, ie., the 20-th frame

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
      ax[0,2].title.set_text('Prediction output')
      ax[0,2].axis('off')

#      qk_[k] = ax_.quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, color=color_map[k])
#      ax_.quiverkey(qk_[k], X=0.0 + k/5.0, Y=1.1, U=20, label=cat_names[k], labelpos='E')
#      ax_.quiver(X_pred, Y_pred, U_pred, V_pred, angles='xy', scale_units='xy', scale=1, width = 0.003, color=color_map[k])
#      ax_.set_xlim(border_pixel, field_pred.shape[0] - border_pixel)
#      ax_.set_ylim(border_pixel, field_pred.shape[1] - border_pixel)
#      ax_.set_aspect('equal')
#      ax_.title.set_text('Prediction output')
#      ax_.axis('off')

    #plt.savefig(os.path.join(img_save_dir, str(frame_idx) + '.png'))
    #plt.imshow(data.item()['motion_pred'].squeeze())
    plt.pause(0.05)
#    ax[0].clear()
#    ax[1].clear()
    ax[0,0].clear()
    ax[0,1].clear()
    ax[0,2].clear()
    ax[1,0].clear()
    ax[1,1].clear()
    ax[1,2].clear()
#    ax_.clear()


def listener():
    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/velodyne_combined", PointCloud2, callback, queue_size=1) # /velodyne_left
    plt.show()
    rospy.spin()

if __name__ == '__main__':
    listener()


