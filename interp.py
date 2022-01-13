from scipy.spatial.transform import Rotation as Rt
import os.path as Pt
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import open3d as o3d
import numpy as np
from copy import deepcopy
import json


def read_anno(file):
    with open(file, 'r') as f:
        cont = json.load(f)
    return cont


def cam2world(rot, trans, outCam):
    se3mat = np.r_[np.c_[rot, trans], [[0.,0.,0.,1.]]]
    true_se3mat = np.linalg.inv(outCam) @ se3mat
    rot = Rt.from_matrix(true_se3mat[:3,:3]).as_rotvec()
    trans = true_se3mat[:,3][:3]
    return np.array(rot, dtype=np.float64), np.array(trans, dtype=np.float64)


def world2cam(rot, trans, outCam):
    se3mat = np.r_[np.c_[rot, trans], [[0.,0.,0.,1.]]]
    true_se3mat = outCam @ se3mat
    rot = Rt.from_matrix(true_se3mat[:3,:3]).as_rotvec()
    trans = true_se3mat[:,3][:3]
    return np.array(rot, dtype=np.float32), np.array(trans, dtype=np.float32)


def interp(func, data, input_ids):
    return func(input_ids, data)(list(range(300)))


def interpolate(in_dict, input_ids=None):
    if input_ids is None:
        input_ids = list(range(0, 295, 10)) + [299]
    rot, trans = in_dict['rot'], in_dict['trans']

    # slerp, interp1d in scipy to interpolate rotation, transition
    rots = interp(Slerp, Rt.from_rotvec(rot), input_ids)
    transis = np.c_[[interp(interp1d, trans[:, i], input_ids) for i in range(3)]].T

    return rots.as_matrix(), transis


def get_all_poses_from_0json_path_and_output_log_path(json_path, out_path):
    json_folder = Pt.dirname(json_path)
    zlen = len(json_path) - len(json_folder) - 6
    outCam = o3d.io.read_pinhole_camera_trajectory(out_path).parameters
    assert len(outCam) == 300
    to_inter = {'rot':[], 'trans':[]}
    all_rots, all_transis = [], []
    for frame_id in range(0, 301, 10):
        if frame_id == 300:
            frame_id = 299
        json_path = Pt.join(json_folder, str(frame_id).zfill(zlen) + '.json')
        obj_data = read_anno(json_path)["dataList"][0]
        trans, rot = obj_data["center"], obj_data["rotation"]
        trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float64)
        rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float64)
        rot = Rt.from_euler('XYZ', rot).as_matrix()
        rot_vec, trans_vec = cam2world(rot, trans, outCam[frame_id].extrinsic)
        to_inter['rot'].append(rot_vec)
        to_inter['trans'].append(trans_vec)
    to_inter['rot'] = np.array(to_inter['rot'])
    to_inter['trans'] = np.array(to_inter['trans'])
    rot_matrix, transis = interpolate(to_inter)
    for frame_id in range(299):
        rot, trans = world2cam(rot_matrix[frame_id], transis[frame_id], outCam[frame_id].extrinsic)
        all_rots.append(rot)
        all_transis.append(trans)
    return all_rots, all_transis


def get_large_gap_poses_from_0json_path_and_output_log_path(json_path, out_path):
    json_folder = Pt.dirname(json_path)
    zlen = len(json_path) - len(json_folder) - 6
    outCam = o3d.io.read_pinhole_camera_trajectory(out_path).parameters
    assert len(outCam) == 300
    to_inter = {'rot':[], 'trans':[]}
    all_rots, all_transis = [], []
    fake_rots, fake_transis = [], []
    for frame_id in range(0, 301, 20):
        if frame_id == 300:
            frame_id = 299
        json_path = Pt.join(json_folder, str(frame_id).zfill(zlen) + '.json')
        obj_data = read_anno(json_path)["dataList"][0]
        trans, rot = obj_data["center"], obj_data["rotation"]
        trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float64)
        rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float64)
        rot = Rt.from_euler('XYZ', rot).as_matrix()
        rot_vec, trans_vec = cam2world(rot, trans, outCam[frame_id].extrinsic)
        to_inter['rot'].append(rot_vec)
        to_inter['trans'].append(trans_vec)
    to_inter['rot'] = np.array(to_inter['rot'])
    to_inter['trans'] = np.array(to_inter['trans'])
    rot_matrix, transis = interpolate(to_inter, input_ids=list(range(0, 299, 20))+[299])
    for frame_id in range(10, 301, 20):
        json_path = Pt.join(json_folder, str(frame_id).zfill(zlen) + '.json')
        obj_data = read_anno(json_path)["dataList"][0]
        trans, rot = obj_data["center"], obj_data["rotation"]
        trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float64)
        rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float64)
        rot = Rt.from_euler('XYZ', rot).as_rotvec()
        all_rots.append(rot)
        all_transis.append(trans)
    for frame_id in range(10, 301, 20):
        rot, trans = world2cam(rot_matrix[frame_id], transis[frame_id], outCam[frame_id].extrinsic)
        fake_rots.append(rot)
        fake_transis.append(trans)
    return all_rots, all_transis, fake_rots, fake_transis