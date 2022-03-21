import open3d as o3d
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rt
from CustomedVisualizer import VisOpen3D
import cv2
import os
from copy import deepcopy


def output_multi_view_images(r, t, pcdList, camIntrinsic, w = 384,
                             h = 384, num = 20, out_name = 'test'):
    """
    r: global rotation in axis-angle
    t: global translation
    pcdList: list of colored open3d.geometry.PointCloud
    camIntrinsic: 3x3 matrix of camera intrinsics
    w, h: size of output image
    num: only 20 + 2 views are supported now, do not change it
    out_name: out_name + _color_ + 0~21.png are the output files
    """
    # t is required to compute camera extrinsic!
    dist = np.linalg.norm(t)
    z_dir = Rt.from_rotvec(r).as_matrix()[:3, 2]
    new_z_vec = t / dist
    new_x_vec = -np.cross(new_z_vec, z_dir)
    new_y_vec = np.cross(new_z_vec, new_x_vec)
    new_x_vec /= - np.linalg.norm(new_x_vec)
    new_y_vec /= - np.linalg.norm(new_y_vec)
    init_ex_mat = np.r_[np.c_[new_x_vec, new_y_vec, new_z_vec, t],
                   [[0,0,0,1]]]

    # dist *= 0.6
    # create window
    vis = VisOpen3D(width=w, height=h, visible=False)

    # point cloud
    for pcd in pcdList:
        vis.add_geometry(pcd)


    # update view

    GR = (1 + math.sqrt(5)) / 2
    DR = 1 / GR
    pts_list = [
        [0, GR, DR],
        [DR, 0, GR],
        [GR, DR, 0],
        [0, -GR, DR],
        [DR, 0, -GR],
        [-GR, DR, 0],
        [1, 1, 1.],
        [1, 1, -1.],
        [1, -1, -1.],
        [-1, 1, -1.]
    ]
    pts = np.r_[[[0., 0., 0.]], np.array(pts_list), -np.array(pts_list)]
    for i in range(0, num + 2):
        # generate multi-view theta, phi
        extrinsic = np.eye(4)
        if 0 < i < num + 1:
            phi = math.atan2(pts[i][1], pts[i][0])
            theta = math.atan2(pts[i][2], np.linalg.norm(pts[i][:2]))
            # print(i, theta, phi)
            rel_rot_mat = Rt.from_euler("xyz", [0., phi, theta]).as_matrix()
            rel_rot_mat[:, 0] = np.cross(rel_rot_mat[:, 2], z_dir)
            rel_rot_mat[:, 0] /= np.linalg.norm(rel_rot_mat[:, 0])
            rel_rot_mat[:, 1] = np.cross(rel_rot_mat[:, 2], rel_rot_mat[:, 0])
            extrinsic[:3, :3] = rel_rot_mat
            extrinsic = init_ex_mat @ extrinsic
            further_trans = extrinsic[:3, 2] * dist
            extrinsic[:3, 3] -= further_trans
            extrinsic = np.linalg.inv(extrinsic)
        elif i == num + 1:
            rel_rot_mat = Rt.from_euler("xyz", [0., 0., 0.]).as_matrix()
            extrinsic[:3, :3] = rel_rot_mat
            extrinsic = init_ex_mat @ extrinsic
            further_trans = extrinsic[:3, 2] * dist
            extrinsic[:3, 3] -= further_trans
            extrinsic = np.linalg.inv(extrinsic)

        vis.update_view_point(camIntrinsic, extrinsic)

        # capture images
        depth = vis.capture_depth_float_buffer(show=False)
        image = vis.capture_screen_float_buffer(show=False)

        vis.capture_screen_image(out_name + "_color_{}.png".format(i))
        # vis.capture_depth_image(out_name + "_depth_{}.png".format(i))

    del vis


def vis_multi_view_object(r, T, camMat, vertices, rgb_path, depth_path, crop_list, out_name):
    """
    object paths -> output_multi_view_images
    """
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_crop = np.zeros_like(depth_raw)
    if crop_list is not None:
        depth_crop[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]] = depth_raw[crop_list[0]:crop_list[1],                                                                 crop_list[2]:crop_list[3]]
    else:
        depth_crop = depth_raw
    depth_raw = o3d.geometry.Image(depth_crop)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False)


    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2]))

    # down_sample
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.001)
    pcd, ind = voxel_down_pcd.remove_radius_outlier(nb_points=350, radius=0.05)

    vertices = o3d.utility.Vector3dVector(vertices)
    cad = o3d.geometry.PointCloud(vertices)
    cad.paint_uniform_color(np.array([0., 1., 0.]))

    R = Rt.from_rotvec(r).as_matrix()
    mesh_mv = deepcopy(cad).translate(T)
    mesh_mv.rotate(R, center=T)
    pcd_list = [pcd, mesh_mv]

    output_multi_view_images(r, T, pcd_list, camMat, out_name = out_name)


if __name__ == "__main__":
    pass
