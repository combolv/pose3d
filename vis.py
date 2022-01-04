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

'''
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
'''

def vis3d(r, T, rgb_path, depth_path, objfile, cam_in_path, crop_list = None, json0_path = None):
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_crop = np.zeros_like(depth_raw)
    if crop_list is not None:
        depth_crop[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]] = depth_raw[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]]
    depth_raw = o3d.geometry.Image(depth_crop)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False )

    camMat = np.load(cam_in_path)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2]))
        

    if os.path.exists('/mnt/8T/kangbo' + objfile + 'v.npy'):
        vertices = np.load('/mnt/8T/kangbo' + objfile + 'v.npy')
        triangles = np.load('/mnt/8T/kangbo' + objfile + 't.npy')
        vertices = o3d.utility.Vector3dVector(vertices)
        triangles = o3d.utility.Vector3iVector(triangles)
        cad = o3d.geometry.TriangleMesh(vertices, triangles)
    else:
        cad = o3d.io.read_triangle_mesh(objfile)

    R = Rt.from_rotvec(r).as_matrix()
    mesh_mv = copy.deepcopy(cad).translate(T)
    mesh_mv.rotate(R, center=T)
    prt_list = [pcd, mesh_mv]

    if json0_path is not None:
        from loadfile import read_rtd
        _, _, D = read_rtd(json0_path)
        box = o3d.geometry.OrientedBoundingBox(center=T, R=R, extent=D)
        pc = box.get_box_points()
        rft = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(
            center=(np.asarray(pc[6]) + np.asarray(pc[4])) / 2,
            R=R, extent=np.array([0.01, 0.01, 0.01])))
        rft.paint_uniform_color(np.array([1., 0., 0.]))
        prt_list.extend([box, rft])

    o3d.visualization.draw_geometries(prt_list)


def vis2d(model_output_depth, model_output_seg, rgb_path, rgb_crop_list, output_file):
    model_output_depth = model_output_depth.detach().cpu().numpy()
    model_output_seg = model_output_seg.detach().cpu().numpy()
    visible_tuples = np.where(model_output_seg > 0.5)
    all_depth = model_output_depth[visible_tuples]
    dep_max, dep_min = np.max(all_depth), np.min(all_depth)
    model_output_depth[visible_tuples] -= dep_min
    model_output_depth /= (dep_max -  dep_min)
    x0, x1, y0, y1 = rgb_crop_list
    rgb_image = cv2.imread(rgb_path)
    rgb_image = rgb_image[x0:x1, y0:y1, :]
    rgb_image[:, :, 2] += np.array((255 - rgb_image[:, :, 2]) * model_output_seg, dtype=np.uint8)
    cv2.imwrite(output_file, rgb_image)


def vis_depth(depth, output_file, crop_list=None):
    if isinstance(depth, str):
        depth = cv2.imread(depth, cv2.IMREAD_UNCHANGED)
    if crop_list is not None:
        depth = depth[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]]
    visible_tuples = np.where(depth > 1)
    all_depth = depth[visible_tuples]
    dep_max, dep_min = np.max(all_depth), np.min(all_depth)
    depth[visible_tuples] -= dep_min
    depth = np.array(depth, dtype=np.float32) * 255
    depth /= (dep_max - dep_min)
    cv2.imwrite(output_file, depth)


