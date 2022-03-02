from scipy.spatial.transform import Rotation as Rt
import numpy as np
import json
import os
import cv2
import open3d as o3d
import pickle

def crop3d(dim, scaling, pcd):
    edge = dim / 2 * scaling
    valid_test = np.all(-edge < pcd, axis=-1) & np.all(pcd < edge, axis=-1)
    return pcd[np.where(valid_test)]


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
        # 'bottleddrinks': '/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/瓶子/',
        # 'Watercup': '/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/马克杯/',
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

    H_idx = file.index("80000") + 5
    try:
        assert file[H_idx] == '2'
    except:
        return ''

    out_file_name = mapping_dict[labels] + str(num).zfill(3) + '.obj'
    if os.path.exists(out_file_name):
        return out_file_name
    else:
        return ''


def read_anno2articulated_objpath(file):
    try:
        with open(file, 'r', errors='ignore') as f:
            cont = f.read()  # .replace(chr(0), '')
        cont = eval(cont)
        dataList = cont["dataList"]
        assert len(dataList) == 2
        labels = dataList[0]['label']
        assert labels == "Laptopdisplay"
    except:
        return None

    N_idx = file.index('N')
    try:
        num = int(file[N_idx+1:N_idx+3])
    except:
        num = int(file[N_idx+1])
    num_str = str(num).zfill(3)
    CAD_part_canonical_path = '/mnt/8T/HOI4D_CAD_Model/part_annotations/laptop/' + num_str + "/objs/new-0-align.obj"
    CAD_base_path = '/mnt/8T/HOI4D_CAD_Model/part_annotations/laptop/' + num_str + '/objs/new-1.obj'
    CAD_part_path = '/mnt/8T/HOI4D_CAD_Model/part_annotations/laptop/' + num_str + '/objs/new-0.obj'

    mob_path = '/mnt/8T/HOI4D_CAD_Model/part_annotations/laptop/' + num_str + "/mobility_v2.json"

    # print(mob_path, os.path.exists(mob_path))
    # input("cont?")

    if os.path.exists(mob_path):
        return (CAD_part_canonical_path, CAD_base_path, CAD_part_path, mob_path)
    else:
        return None


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
    """
    # print('/mnt/8T/kangbo' + filename, os.path.exists('/mnt/8T/kangbo' + filename))
    # rev_file_name = filename.replace("yiqi", "all/20211110")
    # if os.path.exists('/mnt/8T/HOI4D_clean_mask' + rev_file_name):
    #     mask = cv2.imread('/mnt/8T/HOI4D_clean_mask' + rev_file_name)
    #     new_path, denoise = None, False
    # elif denoise:
    #     new_path = '/mnt/8T/HOI4D_clean_mask' + os.path.dirname(filename)
    #     if not os.path.exists(new_path):
    #         os.makedirs(new_path)
    #     mask = cv2.imread(filename)
    # else:
    #     mask = cv2.imread(filename)
    # print("denoise", denoise)
    dx = [-1, -1, -1, 0, 0, 1, 1, 1]
    dy = [-1, 0, 1, 1, -1, -1, 0, 1]
    clean_mask = np.zeros_like(mask)
    if denoise:
        for i in range(w):
            for j in range(h):
                cnt = 0
                val = 0
                flag_no_black_edge = True
                for num in range(8):
                    try:
                        flag_no_black_edge = flag_no_black_edge and (mask[i + dx[num]][j + dy[num]] != [0, 0, 0]).any()
                        cnt += 1 if (mask[i][j][1] == mask[i+dx[num]][j+dy[num]][1] and
                                     mask[i][j][2] == mask[i+dx[num]][j+dy[num]][2]) else 0
                        val += 1
                    except IndexError:
                        continue
                if cnt / val > 0.5 or flag_no_black_edge:
                    clean_mask[i][j] = mask[i][j]
        cv2.imwrite('/mnt/8T/HOI4D_clean_mask' + filename, clean_mask)
    else:
        clean_mask = mask.copy()
    """
    # rev_file_name = filename.replace("yiqi", "all/20211110")
    # assert os.path.exists(filename)
    clean_mask = cv2.imread(filename)
    w, h, _ = clean_mask.shape
    if h != 1920:
        coef = 1.5
        if os.path.exists(filename):
            clean_mask = cv2.imread(filename.replace("yiqi", "all/20211110"))
    else:
        coef = 1
    color_map = get_color_map()
    obj_idx = np.where((clean_mask == color_map[obj_color][::-1]).all(axis=2))
    if obj_color == 3:
        obj_base_idx = np.where((clean_mask == color_map[1][::-1]).all(axis=2))

        obj_idx = (np.c_[[obj_idx[0]], [obj_base_idx[0]]], np.c_[[obj_idx[1]], [obj_base_idx[1]]])


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

    x1,x2,y1,y2=round(crop_min_x * coef)+1, round(crop_max_x * coef)-1,round(crop_min_y * coef)+1, round(crop_max_y * coef)-1
    if x1 <= 0:
        x1 = 0
        if x2 <= 0:
            x2 = 10
    if y1 <= 0:
        y1 = 0

    return int(x1), int(x2), int(y1), int(y2), clean_mask #int(cx), int(cy)  #, d_size * 1.5)


def read_CAD_model_part(path, scale, trans):
    if not os.path.exists(path):
        print(path)
        raise AssertionError
    mesh = o3d.io.read_triangle_mesh(path)
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
    mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                               contraction=o3d.geometry.SimplificationContraction.Average)
    vertices = np.asarray(mesh_smp.vertices)
    triangles = np.asarray(mesh_smp.triangles)
    v_max = np.max(vertices, axis=0)
    v_min = np.min(vertices, axis=0)
    v_center = (v_max + v_min) / 2
    vertices -= v_center
    vertices *= scale
    vertices += v_center + trans
    return vertices, triangles


def read_CAD_model(CAD_path, dim, keep_translation=False, return_translation=False,
                   label=None):
    if os.path.exists('/mnt/8T/kangbo' + CAD_path + 'v.npy'):
        vertices = np.load('/mnt/8T/kangbo' + CAD_path + 'v.npy')
        triangles = np.load('/mnt/8T/kangbo' + CAD_path + 't.npy')
    else:
        # new_path = '/mnt/8T/kangbo' + os.path.dirname(CAD_path)
        # if not os.path.exists(new_path):
        #     os.makedirs(new_path)
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
        if label == 'laptop':
            scalar = scalar[:2]
        max_scalar = np.max(scalar)

        if max_scalar > 1.:
            vertices /= max_scalar
            coef = max_scalar
        elif max_scalar < 0.92:
            vertices /= max_scalar / 0.95
            coef = max_scalar / 0.95
        else:
            coef = 1
        # np.save('/mnt/8T/kangbo' + CAD_path + 'v.npy', vertices)
        # np.save('/mnt/8T/kangbo' + CAD_path + 't.npy', triangles)
        if keep_translation:
            vertices += v_center
        if return_translation:
            return vertices, triangles, v_center, coef
    return vertices, triangles


def folder_path_generator_from_total_json(total_path, articulated=False):
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
        if not articulated:
            CAD_path = read_anno2objpath(json_path + '/' + str(0) + '.json')
            if not CAD_path:
                continue
        else:
            CAD_path = read_anno2articulated_objpath(json_path + '/' + str(0) + '.json')
            if CAD_path is None:
                continue

        yield [cam_in_path, mask_path, cam_out_path, depth_path, rgb_path, json_path, CAD_path]


def annoed_path_generator_from_total_json(total_path, articulated=False):
    for data in folder_path_generator_from_total_json(total_path, articulated):
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


def path_list2plainobj_input(cam_in_path, dpt_path, mask_path, anno_path, CAD_path, out_src_rot=None, outsrc_trans=None,
                             denoise=False):
    rot, trans, dim = read_rtd(anno_path)
    if out_src_rot is not None:
        rot = out_src_rot
    if outsrc_trans is not None:
        trans = outsrc_trans
    vertices, triangles = read_CAD_model(CAD_path, dim)
    depth2d = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
    camMat = np.load(cam_in_path)
    x1, x2, y1, y2, clean_mask = read_mask2bbox(mask_path, denoise=denoise)
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

    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.001)

    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=350, radius=0.05)
    vertices_ = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
    R = Rt.from_rotvec(rot).as_matrix()
    mesh_mv = vertices_.translate(trans)
    mesh_mv.rotate(R, center=trans)
    # o3d.io.write_point_cloud("test2.ply", mesh_mv)
    # o3d.io.write_point_cloud("test.ply", cl)
    # print(len(cl.points))
    # input()
    pcd = np.asarray(cl.points)

    rot_matrix = Rt.from_rotvec(rot).as_matrix()
    pcd = pcd @ rot_matrix - rot_matrix.T @ trans
    pcd = crop3d(dim, 2, pcd)
    pcd = pcd @ rot_matrix.T + trans

    # if len(pcd) < 100:
    #     pcd_show1 = o3d.utility.Vector3dVector(org_pcd)
    #     box = o3d.geometry.OrientedBoundingBox(center=trans, R=Rt.from_rotvec(rot).as_matrix(), extent=dim)
    #     box2 = o3d.geometry.OrientedBoundingBox(center=trans, R=Rt.from_rotvec(rot).as_matrix(), extent=2 * dim)
    #     pcd_show2 = o3d.utility.Vector3dVector(pcd)
    #     o3d.visualization.draw_geometries([o3d.geometry.PointCloud(pcd_show1), box])
    #     o3d.visualization.draw_geometries([o3d.geometry.PointCloud(pcd_show2), box2])

    return [rot, trans, vertices, triangles, obj_mask, hand_mask, depth2d, pcd, camMat, crop_list]


class Axis:
    def __init__(self, path):
        if path is None:
            return
        with open(path, "r") as f:
            res = json.load(f)
            axis_meta = res[0]["jointData"]
            axis_info, limit = axis_meta["axis"], axis_meta["limit"]
            self.orig = np.array(axis_info["origin"], dtype=np.float32)
            self.direction = np.array(axis_info["direction"], dtype=np.float32)
            self.direction /= np.linalg.norm(self.direction)
            t_max, t_min, no_lim = limit["a"], limit["b"], limit["noLimit"]
            self.rad_min = - t_min / 180 * np.pi
            self.rad_max = - t_max / 180 * np.pi
            self.rot_mat = Rt.from_rotvec(self.direction).as_matrix()
            
    def get_relative_rt(self, theta):
        rot_mat = Rt.from_rotvec(self.direction * theta).as_matrix()
        virt_trans = self.orig - rot_mat @ self.orig
        return rot_mat, virt_trans


def path_list2artobj_input(cam_in_path, dpt_path, mask_path, anno_path, CAD_meta,
                           out_src_rot=None, out_src_trans=None,
                           out_src_part_rot=None, out_src_part_trans=None):
    CAD_part_canonical_path, CAD_base_path, CAD_part_path, mob_path = CAD_meta
    rot_base, trans_base, dim_base = read_rtd(anno_path, 0)
    rot_part, trans_part, dim_part = read_rtd(anno_path, 1)
    if out_src_rot is not None:
        rot_base = out_src_rot
    if out_src_trans is not None:
        trans_base = out_src_trans
    if out_src_part_rot is not None:
        rot_part = out_src_part_rot
    if out_src_part_trans is not None:
        trans_part = out_src_part_trans
    theta = np.linalg.norm((Rt.from_rotvec(rot_base) * Rt.from_rotvec(rot_part).inv()).as_rotvec())

    vertices_base, triangles_base, base_translation, coef = read_CAD_model(CAD_base_path, dim_base, return_translation=True,
                                                                           label='laptop')
    vertices_part, triangles_part = read_CAD_model_part(CAD_part_path, coef, base_translation)
    vertices_part += base_translation
    vertices_part_canonical, triangles_part_canonical = read_CAD_model(CAD_part_canonical_path, dim_part,
                                                                       label='laptop')

    depth2d = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
    camMat = np.load(cam_in_path)
    x1, x2, y1, y2, clean_mask = read_mask2bbox(mask_path, obj_color=3)
    crop_list = [x1, x2, y1, y2]

    depth2d = np.array(depth2d, dtype=np.float32) / 1000
    obj_base_mask, hand_mask = read_mask(clean_mask, dscale=1, crop=crop_list)
    large_mask_base, _ = read_mask(clean_mask, dscale=1)
    large_mask_part, _ = read_mask(clean_mask, obj_color=3, dscale=1)
    large_mask_overall = large_mask_part + large_mask_base
    obj_part_mask = large_mask_part[x1:x2, y1:y2, :]
    obj_overall_mask = obj_base_mask + obj_part_mask
    depth3d_base = o3d.geometry.Image(depth2d * large_mask_base[..., 0])
    depth3d_part = o3d.geometry.Image(depth2d * large_mask_part[..., 0])
    depth3d_all = o3d.geometry.Image(depth2d * large_mask_overall[..., 0])
    depth2d = depth2d[x1:x2, y1:y2]
    hand_mask[np.where(depth2d < 0.001)] = 0
    hand_and_part = hand_mask * ( 1 - obj_part_mask )
    hand_and_base = hand_mask * ( 1 - obj_base_mask )

    # load point cloud from depth
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])
    pcd_part = o3d.geometry.PointCloud.create_from_depth_image(depth3d_part, intrinsics, stride=2)
    pcd_base = o3d.geometry.PointCloud.create_from_depth_image(depth3d_base, intrinsics, stride=2)
    pcd_all = o3d.geometry.PointCloud.create_from_depth_image(depth3d_all, intrinsics, stride=2)
    pcd_part = np.asarray(pcd_part.points)
    pcd_base = np.asarray(pcd_base.points)
    pcd_all = np.asarray(pcd_all.points)

    axis_meta = Axis(mob_path)
    axis_meta.orig += base_translation
    para_meta = [rot_base, trans_base, rot_part, trans_part, theta]
    mesh_meta = [vertices_base, triangles_base, vertices_part, triangles_part, vertices_part_canonical,
                 triangles_part_canonical]
    mask_meta = [obj_base_mask, obj_part_mask, obj_overall_mask, hand_and_part, hand_and_base, hand_mask]
    pcd_list = [pcd_base, pcd_part, pcd_all]

    return [axis_meta, para_meta, mesh_meta, mask_meta, depth2d, pcd_list, camMat, crop_list]


def every_path_generator_from_total_json(total_path, required_range=None, articulated=False):
    for data in folder_path_generator_from_total_json(total_path, articulated):
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

def read_mask2bbox_H2O(filename, hand_color=2, h=1980,w=1080): #right 2 left 3
    mask = read_raw_seg(filename, h,w)
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


def read_mask_reverse_H2O(file, hand_color=2, obj_color=1, dscale=1, crop=(0, 1080, 0, 1920),h=1920,w=1080):
    h, w = int(round((crop[3]-crop[2]) / dscale)
               ), int(round((crop[1]-crop[0]) / dscale))
    mask = read_raw_seg(file,h,w)
    # mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]
    # print(mask.shape)

    # [crop[0]:crop[1], crop[2]:crop[3], :]

    hand_idx = np.where((mask == hand_color).all(axis=2))
    obj_idx = np.where((mask == obj_color).all(axis=2))
    hand_mask = np.zeros((w, h, 3), dtype=np.float32)
    obj_mask = np.ones((w, h, 3), dtype=np.float32)
    obj_mask[obj_idx] = [0, 0, 0]
    hand_mask[hand_idx] = [1, 1, 1]### TODO: double hands
    return obj_mask, hand_mask


def loadPickleData(fName):
    with open(fName, 'rb') as f:
        try:
            pickData = pickle.load(f, encoding='latin1')
        except:
            pickData = pickle.load(f)

    return pickData



def read_raw_seg(maskfilepth, h=1920, w=1080):  # right 2 left 3
    color_map = get_color_map()
    mask = cv2.imread(maskfilepth)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (h, w))
    new_mask = np.zeros(mask.shape)
    hand_idx_left = np.where((mask == color_map[3]).all(axis=2))#H2O的时候要反过来
    hand_idx_right = np.where((mask == color_map[2]).all(axis=2))

    new_mask[hand_idx_left] = [3, 3, 3]
    new_mask[hand_idx_right] = [2, 2, 2]
    mask[hand_idx_right] = 255

    # maskrawpth = join(saveRawSegDir, i+'.png')
    # print(hand_idx_right, '!!!')
    # cv2.imwrite('/home/jiangche/HOI4D/masktmp.png', mask)
    return new_mask


def get_wrist_vertline(wrist, mean):
    #y=kx+b, k = (mean[1] - wrist[1])/(mean[0] - wrist[0]), b = wrist[1]-k*wrist[0]
    #b' = wrist[1]
    # k_vert = -(mean[0] - wrist[0])/(mean[1] - wrist[1])
    # b_vert = wrist[1]-k_vert*wrist[0]
    k = (mean[1] - wrist[1])/max(0.001, (mean[0] - wrist[0]))
    b = wrist[1]-k*wrist[0]

    bias = 40
    bias_x = wrist[0]
    bias_y = k*bias_x+b+bias

    k_vert = -(mean[0] - wrist[0])/(mean[1] - wrist[1])
    b_vert = bias_y-k_vert*bias_x

    return k_vert,  b_vert


def read_mask_crop_wrist(file,kps_anno, hand_color=2, obj_color=1, dscale=1, crop=(0, 1080, 0, 1920),h=1920,w=1080):
    h, w = int(round((crop[3]-crop[2]) / dscale)
               ), int(round((crop[1]-crop[0]) / dscale))
    mask = read_raw_seg(file, h, w)
    # mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    mask = mask[crop[0]:crop[1], crop[2]:crop[3], :]
    # print(mask.shape)

    # [crop[0]:crop[1], crop[2]:crop[3], :]

    hand_idx = np.where((mask == hand_color).all(axis=2))
    obj_idx = np.where((mask == obj_color).all(axis=2))
    hand_mask = np.zeros((w, h, 3), dtype=np.float32)
    obj_mask = np.ones((w, h, 3), dtype=np.float32)
    obj_mask[obj_idx] = [0, 0, 0]
    hand_mask[hand_idx] = [1, 1, 1]  # TODO: double hands
    mean = np.mean(kps_anno['coord'], axis=0)
    for kps_coord in kps_anno['coord']:
        cv2.circle(mask, center=(int(kps_coord[0]), int(
            kps_coord[1])), radius=3, color=[255, 255, 0], thickness=-1)
        cv2.circle(mask, center=(int(mean[0]), int(mean[1])), radius=3, color=[
                   255, 0, 255], thickness=-1)
    k, b = get_wrist_vertline(kps_anno['coord'][0], mean)

    for index_cnt in range(len(hand_idx[1])):
        if(k*hand_idx[1][index_cnt]+b < hand_idx[0][index_cnt]):
            mask[hand_idx[0][index_cnt], hand_idx[1][index_cnt]] = [255, 255, 255]
            hand_mask[hand_idx[0][index_cnt],hand_idx[1][index_cnt]] = [0, 0, 0]
    
    cv2.imwrite('/home/jiangche/HOI4D/masktmp.png', mask)
    return obj_mask, hand_mask
    

if __name__ == "__main__":
    # test()
    raise NotImplementedError('You should run main.py instead of loadfile.py')