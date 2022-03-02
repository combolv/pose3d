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
        trans_path = '/nas/HOI4D_ObjPose_Cache/bottle_check/' + str(i) + '/trans.npy'
        rot_path = '/nas/HOI4D_ObjPose_Cache/bottle_check/' + str(i) + '/rot.npy'
        if not (Pt.exists(trans_path) and Pt.exists(rot_path)):
            print(i, 'false')
            continue
        json_folder = Pt.dirname(path_proto[3])
        json_folder_write = json_folder + '_refined'
        # os.mkdir(json_folder_write)
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


if __name__ == "__main__":
    main('bottle_list.pt')