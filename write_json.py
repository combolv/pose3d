from scipy.spatial.transform import Rotation as Rt
import os
import os.path as Pt
import copy
import open3d as o3d
import numpy as np
import json
import torch


def read_anno(file):
    with open(file, 'r') as f:
        cont = json.load(f)
    return cont


def json_proto(file):
    data = read_anno(file)['dataList']
    num_obj = len(data)
    for i in range(num_obj):
        for key in data[i]["2dBox"]:
            data[i]["2dBox"][key] = []
    data_list = list(data)
    ret = [{'frameId': frame, 'filePath': '', 'isEffective': 1, 'dataList': copy.deepcopy(data_list)} for frame in range(300)]
    return ret, num_obj


def main(list_path):
    akoga = torch.load(list_path)
    for i, path300 in enumerate(akoga):
        path_proto = path300[0]
        trans_path = '/nas/HOI4D_ObjPose_Cache/mug_check/' + str(i) + '/trans.npy'
        rot_path = '/nas/HOI4D_ObjPose_Cache/mug_check/' + str(i) + '/rot.npy'
        if not (Pt.exists(trans_path) and Pt.exists(rot_path)):
            # print(i, 'false')
            # print(path_proto)
            continue
        json_folder = Pt.dirname(path_proto[3])
        json_folder_write = json_folder + '_refined'
        os.mkdir(json_folder_write)
        print(json_folder_write)
        # continue
        zlen = 5
        for i in range(10):
            if Pt.exists(Pt.join(json_folder, '0' * i + '.json')):
                zlen = i
                break
        else:
            raise FileNotFoundError('No available json found in the given path: ' + json_folder)

        to_save, _ = json_proto(Pt.join(json_folder, '0' * zlen + '.json'))
        rot_list = np.load(rot_path)
        trans_list = np.load(trans_path)

        for frame_id in range(300):
            rot, trans = rot_list[frame_id], trans_list[frame_id]
            rot = Rt.from_rotvec(rot).as_euler('XYZ')
            #print(rot_dcm[frame_id], rot)
            #print(transis[frame_id], trans)
            to_save[frame_id]["dataList"][0]['rotation']['x'] = float(rot[0])
            to_save[frame_id]["dataList"][0]['rotation']['y'] = float(rot[1])
            to_save[frame_id]["dataList"][0]['rotation']['z'] = float(rot[2])
            to_save[frame_id]["dataList"][0]['center']['x'] = float(trans[0])
            to_save[frame_id]["dataList"][0]['center']['y'] = float(trans[1])
            to_save[frame_id]["dataList"][0]['center']['z'] = float(trans[2])
        for frame_id in range(300):
            # if frame_id % 10 == 0:
            #     continue
            # print(json_folder_write)
            # input()
            with open(Pt.join(json_folder_write, str(frame_id).zfill(4) + '.json'), 'w') as f:
                f.write(json.dumps(to_save[frame_id]))


def tmp_write_json():
    bowl_list = [
        [
            "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N01/S55/s1/T1/",
            "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/001.obj",
            True,
            1
        ],
        [
            "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N05/S55/s3/T1/",
            "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/005.obj",
            True
        ],
        [
            "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N07/S55/s4/T1/",
            "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/007与模型照片不一致.obj",
            False,
        ],
        [
            "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N14/S64/s2/T1/",
            "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/014.obj",
            True
        ],
        [
            "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N26/S79/s5/T1/",
            "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/碗/014.obj",
            True
        ]
    ]
    for k, bowl in enumerate(bowl_list):
        for i in range(1):
            trans_path = '/home/yunze/pose3d_lkb/' + str(k) + 'bowl_check/0/trans.npy'
            rot_path = '/home/yunze/pose3d_lkb/' + str(k) + 'bowl_check/0/rot.npy'
            if not (Pt.exists(trans_path) and Pt.exists(rot_path)):
                print(trans_path)
                print(rot_path)
                continue
            json_folder = Pt.dirname(bowl[0] + 'objpose/')
            json_folder_write = json_folder + '_refined'
            try:
                os.mkdir(json_folder_write)
            except FileExistsError:
                pass
                # print(json_folder_write)
                # break
            # continue
            zlen = 5
            for i in range(10):
                if Pt.exists(Pt.join(json_folder, '0' * i + '.json')):
                    zlen = i
                    break
            else:
                raise FileNotFoundError('No available json found in the given path: ' + json_folder)

            to_save, _ = json_proto(Pt.join(json_folder, '0' * zlen + '.json'))
            rot_list = np.load(rot_path)
            trans_list = np.load(trans_path)
            if len(rot_list) == 300:
                frames = range(300)
            else:
                frames = sum([[j * 10 + i for i in range(1, 10)] for j in range(29)] +
                             [[i for i in range(291, 300)]], [])
                # print(len(frames), len(rot_list))
                if not len(frames) == len(rot_list):
                    print(k, bowl)
                    raise AssertionError
            print('s')
            for d, frame_id in enumerate(frames):
                rot, trans = rot_list[d], trans_list[d]
                rot = Rt.from_rotvec(rot).as_euler('XYZ')
                # print(rot_dcm[frame_id], rot)
                # print(transis[frame_id], trans)
                to_save[frame_id]["dataList"][0]['rotation']['x'] = float(rot[0])
                to_save[frame_id]["dataList"][0]['rotation']['y'] = float(rot[1])
                to_save[frame_id]["dataList"][0]['rotation']['z'] = float(rot[2])
                to_save[frame_id]["dataList"][0]['center']['x'] = float(trans[0])
                to_save[frame_id]["dataList"][0]['center']['y'] = float(trans[1])
                to_save[frame_id]["dataList"][0]['center']['z'] = float(trans[2])
            for frame_id in range(300):
                with open(Pt.join(json_folder_write, str(frame_id).zfill(4) + '.json'), 'w') as f:
                    f.write(json.dumps(to_save[frame_id]))


def write_json_special():
    now_path = "/home/yunze/pose3d_lkb/0bowl_check/0/"
    goal_path = "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C7/N01/S55/s1/T1/",
    frames = frames = sum([[j * 10 + i for i in range(1, 10)] for j in range(29)] +
                             [[i for i in range(291, 300)]], [])
    to_save, _ = json_proto(Pt.join(json_folder, '0' * zlen + '.json'))

if __name__ == "__main__":
    tmp_write_json()