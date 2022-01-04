import open3d as o3d
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rt
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import copy

def read_anno2(file):
    with open(file, 'r') as f:
        cont = f.read()
    cont = eval(cont)
    return len(cont["dataList"])


def read_anno(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
    cont = eval(cont)
    return cont["dataList"][num]


def anno2rt(anno):
    trans, rot, dim = anno["center"], anno["rotation"], anno['dimensions']
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot_raw = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float32)
    dim = np.array([dim['length'], dim['width'], dim['height']])
    rot_dcm = Rt.from_euler('XYZ', rot_raw).as_matrix()
    return trans, rot_dcm, dim

total_json = None

rgb_path = 'E:\\Env\\4dscene\\0.jpg'
depth_path = 'E:\\Env\\4dscene\\0.png'
mask_path = '/home/data/kinectv2/task1/mask/00000.png'
objfile = "E:\\Env\\4dscene\\car.obj"
json_path = 'E:\\Env\\4dscene\\0.json'
cam_in_path = 'E:\\Env\\4dscene\\0.npy'

color_raw = o3d.io.read_image(rgb_path)
depth_raw = o3d.io.read_image(depth_path)

rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, convert_rgb_to_intensity=False )

camMat = np.load(cam_in_path)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2]))
    

cad = o3d.io.read_triangle_mesh(objfile)

T, R, D = anno2rt(read_anno(json_path))
test_box = o3d.geometry.OrientedBoundingBox(center=np.zeros((3,1)), R=np.eye(3),
                extent=np.ones((3,1)))
# pc = test_box.get_box_points()
# print()
# input()
box = o3d.geometry.OrientedBoundingBox(center=T, R=R, extent=D)
pc = box.get_box_points()
rft = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(
        center=(np.asarray(pc[6])+np.asarray(pc[4]))/2,
         R=R, extent=np.array([0.01,0.01,0.01])))
rft.paint_uniform_color(np.array([1.,0.,0.]))
# rft = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([pc[0]]))#, colors=[[0, 1, 0]])
# print(np.asarray(pc)[0])
# input()
mesh_mv = copy.deepcopy(cad).translate(T)
# R = mesh_mv.get_rotation_matrix_from_xyz(rot_raw)
mesh_mv.rotate(R, center=T)

o3d.visualization.draw_geometries([pcd, mesh_mv])


def vis(r, t, rgb, dpt, msk, cam):
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False )

    camMat = np.load(cam_in_path)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image,o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2]))
        

    cad = o3d.io.read_triangle_mesh(objfile)

    T, R, D = anno2rt(read_anno(json_path))
    test_box = o3d.geometry.OrientedBoundingBox(center=np.zeros((3,1)), R=np.eye(3),
                    extent=np.ones((3,1)))
    # pc = test_box.get_box_points()
    # print()
    # input()
    box = o3d.geometry.OrientedBoundingBox(center=T, R=R, extent=D)
    pc = box.get_box_points()
    rft = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(
            center=(np.asarray(pc[6])+np.asarray(pc[4]))/2,
            R=R, extent=np.array([0.01,0.01,0.01])))
    rft.paint_uniform_color(np.array([1.,0.,0.]))
    # rft = o3d.geometry.PointCloud(o3d.utility.Vector3dVector([pc[0]]))#, colors=[[0, 1, 0]])
    # print(np.asarray(pc)[0])
    # input()
    mesh_mv = copy.deepcopy(cad).translate(T)
    # R = mesh_mv.get_rotation_matrix_from_xyz(rot_raw)
    mesh_mv.rotate(R, center=T)

    o3d.visualization.draw_geometries([pcd, mesh_mv])