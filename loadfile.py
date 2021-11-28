from scipy.spatial.transform import Rotation as Rt
import numpy as np
# import os
import cv2
import open3d as o3d


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


def read_mask(file, hand_color=2, obj_color=1, dscale=10, crop=(0,1080,0,1920)):
    color_map = get_color_map()
    h, w = int(round((crop[3]-crop[2]) / dscale)), int(round((crop[1]-crop[0]) / dscale))
    mask = cv2.imread(file)
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


def read_rt(file):
    with open(file, 'r') as f:
        cont = f.read()
    anno = eval(cont)["dataList"][0]
    trans, rot = anno["center"], anno["rotation"]
    trans = np.array([trans['x'], trans['y'], trans['z']])
    rot = np.array([rot['x'], rot['y'], rot['z']])
    rot = Rt.from_euler('XYZ', rot).as_rotvec()
    return np.array(rot, dtype=np.float32), np.array(trans, dtype=np.float32)


# TODO: neat segmentation is needed, maybe some preprocessing when not connected parts
def read_mask2bbox(filename, obj_color=1):
    h, w = 1280, 720
    mask = cv2.imread(filename)
    obj_idx = np.where((mask == color_map[obj_color][::-1]).all(axis=2))
    min_x, max_x = np.min(obj_idx[0]), np.max(obj_idx[0])
    min_y, max_y = np.min(obj_idx[1]), np.max(obj_idx[1])
    x_size = max_x - min_x
    y_size = max_y - min_y
    crop_max_x = min(max_x + x_size / 2, h)
    crop_min_x = max(min_x - x_size / 2, 0)
    crop_max_y = min(max_y + y_size / 2, h)
    crop_min_y = max(min_y - y_size / 2, 0)
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
        # d_size = h / crop_y_size

    x1,x2,y1,y2=round(crop_min_x * 1.5)+1, round(crop_max_x * 1.5)-1,round(crop_min_y * 1.5)+1, round(crop_max_y * 1.5)-1
    cx,cy = -x1, -y1
    return int(x1), int(x2), int(y1), int(y2), int(cx), int(cy)  #, d_size * 1.5)


if __name__ == "__main__":
    raise NotImplementedError('You should run main.py instead of loadfile.py')