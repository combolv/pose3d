from scipy.spatial.transform import Rotation as Rt
import numpy as np
import json
import os
import cv2
import open3d as o3d
import pickle

def get_color_map(N=256):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def read_anno2objpath(file):
    mapping_dict = {
        'bottleddrinks': '/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/瓶子/',
        'Watercup': '/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/马克杯/',
        'Toycar': '/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/玩具车/'
    }
    try:
        with open(file, 'r', errors='ignore') as f:
            cont = f.read()#.replace(chr(0), '')
        cont = eval(cont)
        dataList = cont["dataList"]
        assert len(dataList) == 1
        labels = dataList[0]['label']
    except:
        return ''
    if labels not in mapping_dict:
        return ''
    N_idx = file.index('N')
    try:
        num = int(file[N_idx+1:N_idx+3])
    except:
        num = int(file[N_idx+1])

    out_file_name = mapping_dict[labels] + str(num).zfill(3) + '.obj'

    if os.path.exists(out_file_name):
        return out_file_name
    else:
        return ''


def read_mask(mask, hand_color=2, obj_color=1, dscale=10, crop=(0,1080,0,1920)):
    color_map = get_color_map()
    h, w = int(round((crop[3]-crop[2]) / dscale)), int(round((crop[1]-crop[0]) / dscale))

    mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)[
        crop[0]:crop[1],crop[2]:crop[3],:]
    mask = cv2.resize(mask, (h, w), interpolation=cv2.INTER_NEAREST)
    hand_idx = np.where((mask == color_map[hand_color][::-1]).all(axis=2))
    obj_idx = np.where((mask == color_map[obj_color][::-1]).all(axis=2))
    hand_mask = np.ones((w, h, 3), dtype=np.float32)
    obj_mask = np.zeros((w, h, 3), dtype=np.float32)
    obj_mask[obj_idx] = [1, 1, 1]
    hand_mask[hand_idx] = [0, 0, 0]
    return obj_mask, hand_mask


def read_rt(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
    anno = eval(cont)["dataList"][num]

    trans, rot = anno["center"], anno["rotation"]
    trans = np.array([trans['x'], trans['y'], trans['z']])
    rot = np.array([rot['x'], rot['y'], rot['z']])
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), np.array(trans, dtype=np.float32)


def read_rtd(file, num=0):
    with open(file, 'r') as f:
        cont = f.read()
    anno = eval(cont)["dataList"][num]

    trans, rot, dim = anno["center"], anno["rotation"], anno["dimensions"]
    trans = np.array([trans['x'], trans['y'], trans['z']], dtype=np.float32)
    rot = np.array([rot['x'], rot['y'], rot['z']])
    dim = np.array([dim['length'], dim['width'], dim['height']], dtype=np.float32)
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), trans, dim


def read_mask2bbox(filename, obj_color=1, denoise=True):
    h, w = 1280, 720
    if os.path.exists('/mnt/8T/kangbo' + filename):
        mask = cv2.imread('/mnt/8T/kangbo' + filename)
        new_path, denoise = None, False
    elif denoise:
        new_path = '/mnt/8T/kangbo' + os.path.dirname(filename)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        mask = cv2.imread(filename)
    else:
        mask = cv2.imread(filename)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, 1, -1, -1, 0, 1]
    clean_mask = np.zeros_like(mask)
    if denoise:
        for i in range(w):
            for j in range(h):
                cnt = 0
                val = 0
                for num in range(8):
                    try:
                        cnt += 1 if (mask[i][j][1] == mask[i+dx[num]][j+dy[num]][1] and
                                     mask[i][j][2] == mask[i+dx[num]][j+dy[num]][2]) else 0
                        val += 1
                    except IndexError:
                        continue
                if cnt / val > 0.5:
                    clean_mask[i][j] = mask[i][j]
        cv2.imwrite('/mnt/8T/kangbo' + filename, clean_mask)
    else:
        clean_mask = mask.copy()

    color_map = get_color_map()
    obj_idx = np.where((clean_mask == color_map[obj_color][::-1]).all(axis=2))
    min_x, max_x = np.min(obj_idx[0]), np.max(obj_idx[0])
    min_y, max_y = np.min(obj_idx[1]), np.max(obj_idx[1])
    x_size = max_x - min_x
    y_size = max_y - min_y
    crop_max_x = min(max_x + x_size / 3, w)
    crop_min_x = max(min_x - x_size / 3, 0)
    crop_max_y = min(max_y + y_size / 3, h)
    crop_min_y = max(min_y - y_size / 3, 0)
    crop_y_size = crop_max_y - crop_min_y
    crop_x_size = crop_max_x - crop_min_x
    if crop_x_size > crop_y_size: # y size expand
        if crop_max_y + crop_x_size - crop_y_size > h:
            crop_min_y = crop_max_y - crop_x_size
        else:
            crop_max_y = crop_min_y + crop_x_size
    else: # crop_y_size > crop_x_size, x size expand
        if crop_max_x + crop_y_size - crop_x_size > h:
            crop_min_x = crop_max_x - crop_y_size
        else:
            crop_max_x = crop_min_x + crop_y_size
            if crop_max_x > w:
                crop_min_x -= crop_max_x - w
                crop_max_x -= crop_max_x - w

    x1,x2,y1,y2=round(crop_min_x * 1.5)+1, round(crop_max_x * 1.5)-1,round(crop_min_y * 1.5)+1, round(crop_max_y * 1.5)-1

    return int(x1), int(x2), int(y1), int(y2), clean_mask #int(cx), int(cy)  #, d_size * 1.5)


def read_CAD_model(CAD_path, dim):
    if os.path.exists('/mnt/8T/kangbo' + CAD_path + 'v.npy'):
        vertices = np.load('/mnt/8T/kangbo' + CAD_path + 'v.npy')
        triangles = np.load('/mnt/8T/kangbo' + CAD_path + 't.npy')
    else:
        new_path = '/mnt/8T/kangbo' + os.path.dirname(CAD_path)
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        mesh = o3d.io.read_triangle_mesh(CAD_path)

        # simplify the CAD model
        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
        mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                                   contraction=o3d.geometry.SimplificationContraction.Average)

        # get a bounding box from the CAD model
        vertices = np.asarray(mesh_smp.vertices)
        triangles = np.asarray(mesh_smp.triangles)
        v_max = np.max(vertices, axis=0)
        v_min = np.min(vertices, axis=0)
        v_center = (v_max + v_min) / 2
        v_size = v_max - v_min

        vertices -= v_center
        scalar = v_size / dim
        max_scalar = np.max(scalar)
        if max_scalar > 1.:
            vertices /= max_scalar
        elif max_scalar < 0.92:
            vertices /= max_scalar / 0.95
        np.save('/mnt/8T/kangbo' + CAD_path + 'v.npy', vertices)
        np.save('/mnt/8T/kangbo' + CAD_path + 't.npy', triangles)
    return vertices, triangles


def folder_path_generator_from_total_json(total_path):
    with open(total_path, 'r') as f:
        total_list = f.read()
        total_list = eval(total_list)

    for ptr in range(0, 1034):
        to_check = total_list[ptr]
        if not os.path.exists(to_check[0]):
            raise FileNotFoundError
        cam_in_path = './0.npy'
        for i in range(1, 5):
            if '80000' + str(i) in to_check[0]:
                cam_in_path = './{}.npy'.format(i)
        try:
            mask_path, out_path, depth_path, _, json_path, _ = to_check
        except ValueError:
            continue

        if out_path == '/mnt/8T/HOI4D_data_all/20211110/ZY20210800002/H2/C5/N06/S122/s3/T1/3Dseg':
            # this file is somehow broken
            continue

        mask_path =  mask_path + '/mask/'
        cam_out_path = out_path + '/output.log'
        rgb_path = depth_path[:-5] + 'image'
        CAD_path = read_anno2objpath(json_path + '/' + str(0) + '.json')
        if not CAD_path:
            continue

        yield [cam_in_path, mask_path, cam_out_path, depth_path, rgb_path, json_path, CAD_path]


def annoed_path_generator_from_total_json(total_path):
    for data in folder_path_generator_from_total_json(total_path):
        cam_in_path, mask_path, cam_out_path, depth_path, rgb_path, json_path, CAD_path = data

        all_num_list = list(range(0, 295, 10)) + [299]
        all_ret_list = []

        for num in all_num_list:
            cur_mask_path = mask_path + str(num).zfill(5) + '.png'
            cur_dpt_path = depth_path + '/' + str(num) + '.png'
            cur_json_path = json_path + '/' + str(num) + '.json'
            cur_rgb_path = rgb_path + '/' + str(num) + '.jpg'

            if not os.path.exists(cur_json_path):
                cur_json_path = json_path + '/' + str(num).zfill(5) + '.json'

            all_ret_list.append([cam_in_path,
                                 cur_dpt_path,
                                 cur_mask_path,
                                 cur_json_path,
                                 CAD_path,
                                 cam_out_path,
                                 cur_rgb_path,
                                 ])
        yield all_ret_list


def path_list2plainobj_input(cam_in_path, dpt_path, mask_path, anno_path, CAD_path, *args):
    rot, trans, dim = read_rtd(anno_path)
    vertices, triangles = read_CAD_model(CAD_path, dim)
    depth2d = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
    camMat = np.load(cam_in_path)
    x1, x2, y1, y2, clean_mask = read_mask2bbox(mask_path)
    crop_list = [x1, x2, y1, y2]

    depth2d = np.array(depth2d, dtype=np.float32) / 1000
    obj_mask, hand_mask = read_mask(clean_mask, dscale=1, crop=crop_list)
    large_mask, _ = read_mask(clean_mask, dscale=1)
    depth3d = o3d.geometry.Image(depth2d * large_mask[..., 0])
    depth2d = depth2d[x1:x2, y1:y2]
    hand_mask[np.where(depth2d < 0.001)] = 0

    # load point cloud from depth
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d, intrinsics, stride=2)

    pcd = np.asarray(pcd.points)

    return [rot, trans, vertices, triangles, obj_mask, hand_mask, depth2d, pcd, camMat, crop_list]


def every_path_generator_from_total_json(total_path, required_range=None):
    for data in folder_path_generator_from_total_json(total_path):
        cam_in_path, mask_path, cam_out_path, depth_path, rgb_path, json_path, CAD_path = data

        all_ret_list = []

        if required_range is None:
            required_range = range(300)

        for num in required_range:
            cur_mask_path = mask_path + str(num).zfill(5) + '.png'
            cur_dpt_path = depth_path + '/' + str(num) + '.png'
            cur_json_path = json_path + '/' + '0' + '.json'
            cur_rgb_path = rgb_path + '/' + str(num) + '.jpg'

            if not os.path.exists(cur_json_path):
                cur_json_path = json_path + '/' + '0'.zfill(5) + '.json'

            all_ret_list.append([cam_in_path,
                                 cur_dpt_path,
                                 cur_mask_path,
                                 cur_json_path,
                                 CAD_path,
                                 cam_out_path,
                                 cur_rgb_path,
                                 ])
        yield all_ret_list


def parse_json(file_pth):  # one hand only
    json_pth = file_pth
    with open(json_pth) as j:
        dic = json.load(j)
        kps_cnt = len(dic['markResult']['objects'][0]['features'])
        kps_coord = []
        kps_label = []
        kps_vis = []
        for i in range(kps_cnt):
            kps_coord.append(dic['markResult']['objects']
                             [0]['features'][i]['geometry']['coordinates'])
            kps_label.append(
                int(dic['markResult']['objects'][0]['features'][i]['properties']['label']))
            kps_vis.append(dic['markResult']['objects'][0]['features']
                           [i]['properties']['content']['Occlude'] == 'visible')

    return {'coord': kps_coord,
            'label': kps_label,
            'vis': kps_vis,
            'num': kps_cnt
            }


def read_mask2bbox_hand(filename, hand_color=2, h=1980,w=1080): #right 2 left 3
    mask = cv2.imread(filename)
    hand_idx = np.where((mask == hand_color).all(axis=2))
    # print(hand_idx)
    # newmask = np.zeros((1080,1920,3))
    # newmask[hand_idx] = [255,0,0]
    # cv2.imwrite('mask.png', newmask)
    min_x, max_x = np.min(hand_idx[0]), np.max(hand_idx[0])
    min_y, max_y = np.min(hand_idx[1]), np.max(hand_idx[1])
    # print(min_x, max_x, min_y, max_y)

    x_size = max_x - min_x
    y_size = max_y - min_y
    crop_max_x = min(max_x + x_size / 3, w)
    crop_min_x = max(min_x - x_size / 3, 0)
    crop_max_y = min(max_y + y_size / 3, h)
    crop_min_y = max(min_y - y_size / 3, 0)
    crop_y_size = crop_max_y - crop_min_y
    crop_x_size = crop_max_x - crop_min_x
    if crop_x_size > crop_y_size:  # y size expand
        if crop_max_y + crop_x_size - crop_y_size > h:
            crop_min_y = crop_max_y - crop_x_size
        else:
            crop_max_y = crop_min_y + crop_x_size
    else:  # crop_y_size > crop_x_size, x size expand
        if crop_max_x + crop_y_size - crop_x_size > h:
            crop_min_x = crop_max_x - crop_y_size
        else:
            crop_max_x = crop_min_x + crop_y_size
            if crop_max_x > w:
                crop_min_x -= crop_max_x - w
                crop_max_x -= crop_max_x - w
    x1, x2, y1, y2 = round(crop_min_x * 1.5)+1, round(crop_max_x * 1.5) - \
        1, round(crop_min_y * 1.5)+1, round(crop_max_y * 1.5)-1
    return int(crop_min_x), int(crop_max_x), int(crop_min_y), int(crop_max_y)


def read_mask_reverse(file, hand_color=2, obj_color=1, dscale=10, crop=(0, 1080, 0, 1920)):
    color_map = get_color_map()
    h, w = int(round((crop[3]-crop[2]) / dscale)
               ), int(round((crop[1]-crop[0]) / dscale))
    mask = cv2.imread(file)
    # mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]
    # print(mask.shape)

    # [crop[0]:crop[1], crop[2]:crop[3], :]

    hand_idx = np.where((mask == hand_color).all(axis=2))
    obj_idx = np.where((mask == obj_color).all(axis=2))
    hand_mask = np.zeros((w, h, 3), dtype=np.float32)
    obj_mask = np.ones((w, h, 3), dtype=np.float32)
    obj_mask[obj_idx] = [0, 0, 0]
    hand_mask[hand_idx] = [1, 1, 1]
    return obj_mask, hand_mask


def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData

if __name__ == "__main__":
    # test()
    raise NotImplementedError('You should run main.py instead of loadfile.py')