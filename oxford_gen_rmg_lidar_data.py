# python3

import rospy
from sensor_msgs.msg import PointCloud2
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs import point_cloud2

data_save_dir = '/mnt/Disk1/oxford_rmg_lidar_combined_ts_from_radar_0.0432/2019-01-10-11-46-21-radar-oxford-10k/'

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

voxel_size = (0.25, 0.25, 5.0)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])

voxel_size = (0.25, 0.25, 1.0)
area_extents = np.array([[-32., 32.], [-32., 32.], [-10., 10.]])

#voxel_size = (0.125, 0.125, 5.0)
#area_extents = np.array([[-64., 64.], [-64., 64.], [-3., 2.]])
global old_t
old_t = 0

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


def callback(data):
    #print(data.header.stamp.to_sec())
    file_name = format(data.header.stamp.to_sec(), '.7f')
    #print(file_name)
    global old_t
    if float(file_name) - old_t > 0.09:
      print('!!!')
      print(float(file_name) - old_t)

    pc = np.array(list(point_cloud2.read_points(data, skip_nans=True)))
    #print(pc.shape)

    padded_voxel_points, voxel_indices = voxelize_occupy(pc, voxel_size=voxel_size, extents=area_extents, return_indices=True)
    #print(padded_voxel_points.shape)

    non_empty_map = padded_voxel_points[:,:,:].any(axis=2)
    #print(non_empty_map.shape)

    ### save result ###
    save_data_dict = dict()
    save_data_dict['non_empty_map'] = non_empty_map
#    np.save(data_save_dir+str(file_name)+'.npy', save_data_dict)
#    f = open(data_save_dir+'combined_lidar.timestamps', 'a')
#    f.write(file_name+'\n')
#    f.close()

    ax.imshow(np.flip(non_empty_map, axis=0), cmap = 'binary')
    ax.axis('off')
    ax.set_aspect('equal')
    #ax.title.set_text('vel_map')

    plt.pause(0.005)
    ax.clear()
    old_t = float(file_name)

def listener():

    rospy.init_node('listener', anonymous=True)
    rospy.Subscriber("/velodyne_combined", PointCloud2, callback, queue_size=100000)
    plt.show()
    rospy.spin()

if __name__ == '__main__':
    listener()

