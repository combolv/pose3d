from scipy.spatial.transform import Rotation as Rt
import os.path as Pt
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
import open3d as o3d
import numpy as np
from copy import deepcopy
import json
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam-out", default='./output.log', type=str)
    parser.add_argument("--json-folder", default='./objpose/', type=str)
    args = parser.parse_args()
    return args


def read_anno(file):
    with open(file, 'r') as f:
        cont = json.load(f)
    return cont


def cam2world(rot, trans, outCam):
    se3mat = np.r_[np.c_[rot, trans], [[0.,0.,0.,1.]]]
    true_se3mat = np.linalg.inv(outCam) @ se3mat
    rot = Rt.from_matrix(true_se3mat[:3,:3]).as_euler('XYZ')
    trans = true_se3mat[:,3][:3]
    return np.array(rot, dtype=np.float64), np.array(trans, dtype=np.float64)


def world2cam(rot, trans, outCam):
    se3mat = np.r_[np.c_[rot, trans], [[0.,0.,0.,1.]]]
    true_se3mat = outCam @ se3mat
    rot = Rt.from_matrix(true_se3mat[:3,:3]).as_euler('XYZ')
    trans = true_se3mat[:,3][:3]
    return np.array(rot, dtype=np.float64), np.array(trans, dtype=np.float64)


def interp(func, data):
    return func(list(range(0, 295, 20)), data)(list(range(0, 285, 10)))


def interpolate(in_dict):
    rot, trans = np.array(in_dict['rot'][::2]), np.array(in_dict['trans'][::2])
    all_rot, all_trans = np.array(in_dict['rot']), np.array(in_dict['trans'])

    # slerp, interp1d in scipy to interpolate rotation, transition
    rots = interp(Slerp, Rt.from_euler('XYZ', rot))
    transis = np.c_[[interp(interp1d, trans[:, i]) for i in range(3)]].T

    res = (Rt.from_euler('XYZ', all_rot).inv() * rots).as_rotvec()
    out = [0] * len(res)
    for i in range(len(res)):
        out[i] = np.linalg.norm(res[i])
    
    print(sum(out) / len(out) * 180 / np.pi, max(out) * 180 / np.pi)
    res = transis - all_trans
    out = [0] * len(res)
    for i in range(len(res)):
        out[i] = np.linalg.norm(res[i])
    
    print(sum(out) / len(out), max(out))
    input()
    return None


def json_proto(file):
    data = read_anno(file)['dataList']
    num_obj = len(data)
    for i in range(num_obj):
        for key in data[i]["2dBox"]:
            data[i]["2dBox"][key] = []
    ret = [{
        'frameId' : frame,
        'filePath' : '',
        'isEffective': 1,
        'dataList': deepcopy(data)
    } for frame in range(300)]
    return ret, num_obj

args = get_args()
to_inter = {'rot':[], 'trans':[]}
outCam = o3d.io.read_pinhole_camera_trajectory(args.cam_out).parameters
for frame_id in range(0, 285, 10):
    json_path = Pt.join(args.json_folder, str(frame_id).zfill(5) + '.json')
    obj_data = read_anno(json_path)["dataList"][0]
    trans, rot = obj_data["center"], obj_data["rotation"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float64)
    rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float64)
    rot = Rt.from_euler('XYZ', rot).as_matrix()
    rot_vec, trans_vec = cam2world(rot, trans, outCam[frame_id].extrinsic)
    to_inter['rot'].append(rot_vec)
    to_inter['trans'].append(trans_vec)

rot_matrix, transis = interpolate(to_inter)


# if __name__ == '__main__':
#     args = get_args()
#     outCam = o3d.io.read_pinhole_camera_trajectory(args.cam_out).parameters
#     if len(outCam) != 300:
#         raise FileNotFoundError('Invalid camera extrinsics log: '+args.cam_out)
#     zlen = 5
#     for i in range(10):
#         if Pt.exists(Pt.join(args.json_folder, '0'*i+'.json')):
#             zlen = i
#             break
#     else:
#         raise FileNotFoundError('No available json found in the given path: '+args.json_folder)
#     to_save, num_obj = json_proto(Pt.join(args.json_folder, '0'*zlen+'.json'))
#     for num in range(num_obj):
#         to_inter = {'rot':[], 'trans':[]}
#         for frame_id in range(0, 301, 10):
#             if frame_id == 300:
#                 frame_id = 299
#             json_path = Pt.join(args.json_folder, str(frame_id).zfill(zlen) + '.json')
#             obj_data = read_anno(json_path)["dataList"][num]
#             trans, rot = obj_data["center"], obj_data["rotation"]
#             trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float64)
#             rot = np.array([rot['x'], rot['y'], rot['z']], dtype=np.float64)
#             rot = Rt.from_euler('XYZ', rot).as_matrix()
#             rot_vec, trans_vec = cam2world(rot, trans, outCam[frame_id].extrinsic)
#             to_inter['rot'].append(rot_vec)
#             to_inter['trans'].append(trans_vec)
#         rot_matrix, transis = interpolate(to_inter)
#         for frame_id in range(299):
#             if frame_id % 10 == 0:
#                 continue
#             rot, trans = world2cam(rot_matrix[frame_id], transis[frame_id], outCam[frame_id].extrinsic)
#             to_save[frame_id]["dataList"][num]['rotation']['x'] = float(rot[0])
#             to_save[frame_id]["dataList"][num]['rotation']['y'] = float(rot[1])
#             to_save[frame_id]["dataList"][num]['rotation']['z'] = float(rot[2])
#             to_save[frame_id]["dataList"][num]['center']['x'] = float(trans[0])
#             to_save[frame_id]["dataList"][num]['center']['y'] = float(trans[1])
#             to_save[frame_id]["dataList"][num]['center']['z'] = float(trans[2])
#     for frame_id in range(299):
#         if frame_id % 10 == 0:
#             continue
#         with open(Pt.join(args.json_folder, str(frame_id).zfill(zlen) + '.json'), 'w') as f:
#             f.write(json.dumps(to_save[frame_id]))