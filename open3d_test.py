# Test on ubuntu18 open3d 0.12 (registration_ransac_based_on_correspondence is incorrect in prev. version)
# Conclusion:
# Open3d RANSAC is working, but result.correspondence_set only shows the inliers after distance threshold instead of whole ransac process.

import open3d as o3d
import numpy as np
import copy
import time

def rigid_transform_3D(A, B, scale):
    np.set_printoptions(precision=2)
    assert len(A) == len(B)
    N = A.shape[0];  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    # dot is matrix multiplication for array
    if scale:
      H = np.transpose(BB).dot(AA) / N
    else:
      H = np.transpose(BB).dot(AA)

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = Vt.T.dot(U.T)
    if scale:
        varA = np.var(A, axis=0).sum()
        c = 1 / (1 / varA * np.sum(S))  # scale factor
        t = -R * (centroid_B.T * c) + centroid_A.T
    else:
        c = 1
        t = -R.dot(centroid_B.T) + centroid_A.T
    return c, R, t

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


NOISE_BOUND = 0.0
N_OUTLIERS = 0
OUTLIER_TRANSLATION_LB = 5
OUTLIER_TRANSLATION_UB = 4.5

bun_cloud = o3d.io.read_point_cloud("../TEASER-plusplus/examples/example_data/bun_zipper_res3.ply") # /hostroot

src_cloud = bun_cloud
src_cloud = src_cloud.voxel_down_sample(voxel_size=0.01)

src = np.transpose(np.asarray(src_cloud.points))
N = src.shape[1]
print('N:',N)

# Apply arbitrary scale, translation and rotation
#T = np.array(
#    [[9.96926560e-01, 6.68735757e-02, -4.06664421e-02, -1.15576939e-01],
#     [-6.61289946e-02, 9.97617877e-01, 1.94008687e-02, -3.87705398e-02],
#     [4.18675510e-02, -1.66517807e-02, 9.98977765e-01, 1.14874890e-01],
#     [0, 0, 0, 1]])

#    T = np.array(
#        [[  0.0000000,  0.0000000,  1.0000000, 0],
#         [ 1.0000000,  0.0000000, -0.0000000, 0],
#         [-0.0000000,  1.0000000,  0.0000000, 0],
#         [0, 0, 0, 1]])

T = np.identity(4)
T[0,3] = 5

dst_cloud = copy.deepcopy(src_cloud)
dst_cloud.paint_uniform_color([0, 1, 0])
dst_cloud.transform(T)
dst = np.transpose(np.asarray(dst_cloud.points))

# Add some noise
dst += (np.random.rand(3, N) - 0.5) * 2 * NOISE_BOUND

# Add some outliers
outlier_indices = np.random.randint(N_OUTLIERS, size=N_OUTLIERS)
for i in range(outlier_indices.size):
    shift = OUTLIER_TRANSLATION_LB + np.random.rand(3, 1) * (OUTLIER_TRANSLATION_UB - OUTLIER_TRANSLATION_LB)
    dst[:, outlier_indices[i]] += shift.squeeze()


src_cloud.points = o3d.utility.Vector3dVector(src.T)
dst_cloud.points = o3d.utility.Vector3dVector(dst.T)
src_cloud.paint_uniform_color([1, 0, 0])
dst_cloud.paint_uniform_color([0, 1, 0])
#o3d.visualization.draw_geometries([src_cloud+dst_cloud])

src_f = o3d.registration.Feature()
src_f.data = src
dst_f = o3d.registration.Feature()
dst_f.data = dst

arr = np.expand_dims(np.arange(src.shape[1]),axis=0)
arr_ = np.concatenate((arr, arr), axis=0)
corres = o3d.utility.Vector2iVector(arr_.T)

line_set = gen_corr_line_set(src,dst,corres,[0,0,1])
o3d.visualization.draw_geometries([src_cloud+dst_cloud]+[line_set])

s, R, T = rigid_transform_3D(src.T, dst.T, False)
np.set_printoptions(precision=2)
print(R, T)

#distance_threshold=100
#result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
#        src_cloud, dst_cloud, corres, distance_threshold,
#        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 6,#[],
#        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9)],
#        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500)) #4000000, 500

#result = o3d.registration.registration_ransac_based_on_feature_matching(
#        src_cloud, dst_cloud, src_f, dst_f, distance_threshold,
#        o3d.registration.TransformationEstimationPointToPoint(False), 10,# [],
#        [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.01)],
#        o3d.registration.RANSACConvergenceCriteria(4000000, 500)) #4000000, 500
#print(result.transformation)



#corres_result = result.correspondence_set
#print(corres.count)
#print(corres_result.count)
#line_set = gen_corr_line_set(src, dst, corres_result, [0,0,1])
#o3d.visualization.draw_geometries([src_cloud+dst_cloud]+[line_set])
