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
import math


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
    '''
    input:
        rendered dpt, seg cropped
        rgb croplist 1920*1080
    '''

    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    model_output_depth = model_output_depth.detach().cpu().numpy()
    model_output_seg = model_output_seg.detach().cpu().numpy()
    visible_pixels = np.where(model_output_seg > 0.5)
    all_depth = model_output_depth[visible_pixels]
    dep_max, dep_min = np.max(all_depth), np.min(all_depth)
    model_output_depth[visible_pixels] -= dep_min
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
    visible_pixels = np.where(depth > 1)
    all_depth = depth[visible_pixels]
    dep_max, dep_min = np.max(all_depth), np.min(all_depth)
    depth[visible_pixels] -= dep_min
    depth = np.array(depth, dtype=np.float32) * 255
    depth /= (dep_max - dep_min)
    cv2.imwrite(output_file, depth)


def vis3d_art():
    pass


def showAnnoBox(jsonPathOrMeta, rgb_path, cam_path, out_file):
    if isinstance(jsonPathOrMeta, str):
        from loadfile import read_rtd
        rot, trans, dim = read_rtd(jsonPathOrMeta)
    else:
        rot, trans, dim = jsonPathOrMeta
    x, y, z = dim
    org_pts = [[0, 0, 0], [x, y, z],
               [0, 0, z], [x, y, 0],
               [0, y, 0], [x, 0, z],
               [x, 0, 0], [0, y, z]]
    line_pairs = [[0, 2], [0, 4], [0, 6],
                  [1, 3], [1, 5], [1, 7],
                  [2, 5], [2, 7],
                  [4, 3], [4, 7],
                  [6, 3], [6, 5]]
    org_pts = np.array(org_pts, dtype=np.float32)
    org_pts -= dim / 2
    rot_mat = Rt.from_rotvec(rot).as_matrix().T
    org_pts = org_pts @ rot_mat + trans
    camMat = np.load(cam_path)
    org_pts = org_pts @ camMat.T
    org_pts[..., 0] /= org_pts[..., 2]
    org_pts[..., 1] /= org_pts[..., 2]
    rgb = cv2.imread(rgb_path)
    for line_pair in line_pairs:
        start_point = org_pts[line_pair[0]][:2]
        end_point = org_pts[line_pair[1]][:2]
        green = (0, 255, 0)
        start_point = start_point.astype(np.int)
        end_point = end_point.astype(np.int)
        cv2.line(rgb, start_point, end_point, green, thickness=5)
    cv2.imwrite(out_file, rgb)


def showHandJoints(imgInOrg, gtIn, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying hand annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''

    imgIn = np.copy(imgInOrg)

    # Set color for each finger
    joint_color_code = [[139, 53, 255],
                        [0, 56, 255],
                        [43, 140, 237],
                        [37, 168, 36],
                        [147, 147, 0],
                        [70, 17, 145]]

    limbs = [[0, 1],
             [1, 2],
             [2, 3],
             [3, 4],
             [0, 5],
             [5, 6],
             [6, 7],
             [7, 8],
             [0, 9],
             [9, 10],
             [10, 11],
             [11, 12],
             [0, 13],
             [13, 14],
             [14, 15],
             [15, 16],
             [0, 17],
             [17, 18],
             [18, 19],
             [19, 20]
             ]

    PYTHON_VERSION = 3

    for joint_num in range(gtIn.shape[0]):

        color_code_num = (joint_num // 4)
        if joint_num in [0, 4, 8, 12, 16]:
            if PYTHON_VERSION == 3:
                joint_color = list(
                    map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num %
                                  4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(
                gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)
        else:
            if PYTHON_VERSION == 3:
                joint_color = list(
                    map(lambda x: x + 35 * (joint_num % 4), joint_color_code[color_code_num]))
            else:
                joint_color = map(lambda x: x + 35 * (joint_num %
                                  4), joint_color_code[color_code_num])

            cv2.circle(imgIn, center=(
                gtIn[joint_num][0], gtIn[joint_num][1]), radius=3, color=joint_color, thickness=-1)

    for limb_num in range(len(limbs)):

        x1 = gtIn[limbs[limb_num][0], 1]
        y1 = gtIn[limbs[limb_num][0], 0]
        x2 = gtIn[limbs[limb_num][1], 1]
        y2 = gtIn[limbs[limb_num][1], 0]
        length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if length < 150 and length > 5:
            deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
            polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                       (int(length / 2), 3),
                                       int(deg),
                                       0, 360, 1)
            color_code_num = limb_num // 4
            if PYTHON_VERSION == 3:
                limb_color = list(
                    map(lambda x: x + 35 * (limb_num % 4), joint_color_code[color_code_num]))
            else:
                limb_color = map(lambda x: x + 35 * (limb_num %
                                 4), joint_color_code[color_code_num])

            cv2.fillConvexPoly(imgIn, polygon, color=limb_color)

    if filename is not None:
        cv2.imwrite(filename, imgIn)
    else:
        cv2.imshow('kp+rgb',imgIn)
        cv2.waitKey(0)


    return imgIn

def vis_model_dpt(depth_path, rendered_depth, output_file, crop_list):
    '''
    input:
        rendered dpt, seg cropped
        croplist 1920*1080
    '''
    rendered_depth = rendered_depth.detach().cpu().numpy() * 1000
    # np.where(rendered_depth > 1)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth = depth[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]]
    visible_pixels = np.where(depth > 1)
    all_depth = depth[visible_pixels]
    dep_max, dep_min = np.max(all_depth), np.min(all_depth)
    rendered_depth[visible_pixels] -= dep_min
    rendered_depth = np.array(rendered_depth, dtype=np.float32) * 255
    rendered_depth /= dep_max - dep_min
    cv2.imwrite(output_file, rendered_depth)



