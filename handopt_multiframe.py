import cv2
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from loadfile import *
from vis import *
from random import choice
from model import HandObj
from tqdm import tqdm
import argparse
from os.path import join
import random
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='cuda:0',
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--id', type=str, default='299', help='image id')
FLAGS = parser.parse_args()

jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

DATA_PTH = '/mnt/8T/HOI4D_data_all/20211110'
INIT_PTH = '/mnt/8T/HOI4D_data_all/C1C5'
OUT_PTH = '/mnt/8T/HOI4D_data_all/handposeC1C5_all'


def run():
    cams = os.listdir(INIT_PTH)
    cams = [cam for cam in cams if 'ZY' in cam]
    cams = sorted(cams)
    for cam in cams:
        camdir = join(INIT_PTH, cam)
        if not os.path.exists(join(OUT_PTH, cam)):
            os.mkdir(join(OUT_PTH, cam))
        for staff in os.listdir(camdir):
            staffdir = join(camdir, staff)
            if not os.path.exists(join(OUT_PTH, cam, staff)):
                os.mkdir(join(OUT_PTH, cam, staff))
            for cat in os.listdir(staffdir):
                catdir = join(staffdir, cat)
                if not os.path.exists(join(OUT_PTH, cam, staff, cat)):
                    os.mkdir(join(OUT_PTH, cam, staff, cat))
                for obj in sorted(os.listdir(catdir)):
                    objdir = join(catdir, obj)
                    if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj)):
                        os.mkdir(join(OUT_PTH, cam, staff, cat, obj))
                    for scene in os.listdir(objdir):
                        scenedir = join(objdir, scene)
                        if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene)):
                            os.mkdir(join(OUT_PTH, cam, staff, cat, obj, scene))
                        for setting in os.listdir(scenedir):
                            settingdir = join(scenedir, setting)
                            if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene, setting)):
                                os.mkdir(join(OUT_PTH, cam, staff, cat, obj, scene, setting))
                            for task in os.listdir(settingdir):
                                taskdir = join(settingdir, task)
                                if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene, setting, task)):
                                    os.mkdir(join(OUT_PTH, cam, staff, cat, obj, scene, setting, task))

                                if (cat == 'C5'):
                                    print(datetime.datetime.now())
                                    print('data:',taskdir)
                                    image_ids = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150,
                                                156, 162, 168, 174, 180, 186, 192, 198, 204, 210, 216, 222, 228, 234, 240, 246, 252, 258, 264, 270, 276, 282, 288, 294, 299]
                                    for image_id in image_ids:
                                        image_id = str(image_id)
                                        outdir = join(
                                            OUT_PTH, cam, staff, cat, obj, scene, setting, task, image_id)
                                        if not os.path.exists(outdir):
                                            os.mkdir(outdir)

                                        datadir = join(DATA_PTH, cam, staff, cat, obj, scene, setting, task)
                                        initdir = join(INIT_PTH, cam, staff, cat, obj, scene, setting, task)

                                        depth2d_pth = datadir+'/align_depth/'+image_id+'.png'
                                        color2d_pth = datadir+'/align_image/'+image_id+'.jpg'
                                        hand_anno_pth = datadir+'/handpose/'+image_id+'.json'
                                        mask_pth = initdir+'/segmentation/raw_seg_results/'+image_id+'.png'
                                        pkl_pth = initdir+'/handInit/singleFrameFit/'+image_id+'.pickle'
                                        camparam_pth = INIT_PTH+'/camera_params/'+cam+'/color_intrin.npy'

                                        if not os.path.exists(pkl_pth):
                                            print(pkl_pth,' do not exist')
                                            continue
                                        if os.path.exists(join(outdir, 'newhand.pickle')):
                                            print(join(outdir, 'newhand.pickle'), ' already exist')
                                            continue

                                        #get color
                                        color = cv2.imread(color2d_pth)

                                        #get camMat
                                        camMat = np.load(camparam_pth)

                                        #get 2d depth
                                        depth2d = cv2.imread(depth2d_pth, cv2.IMREAD_UNCHANGED)
                                        depth2d = np.array(depth2d, dtype=np.float32) / 1000  # mm

                                        #get anno 2d keypoints and their visibility
                                        kps_anno = parse_json(hand_anno_pth)
                                        cur_kps2D = np.zeros((21, 2))
                                        cur_vis = np.zeros((21, 1))
                                        for kps_i in range(kps_anno['num']):
                                            assert kps_i < 21, 'too many kps!!'
                                            idx = jointsMap.index(kps_anno['label'][kps_i])
                                            cur_kps2D[idx] = kps_anno['coord'][kps_i]
                                            if kps_anno['vis'][kps_i] == True:
                                                cur_vis[idx] = 1.0
                                            else:
                                                cur_vis[idx] = 0.8
                                        cur_vis[0] = 5.0  # !!!

                                        x1, x2, y1, y2 = read_mask2bbox_hand(mask_pth)
                                        crop_list = [x1, x2, y1, y2]  # 1920*1080 square
                                        # print(crop_list,'croplist')
                                        obj_mask, hand_mask = read_mask_reverse(mask_pth, dscale=1, crop=crop_list)
                                        large_obj_mask, large_hand_mask = read_mask_reverse(mask_pth, dscale=1)
                                        # ['KPS2D', 'imgID', 'conf', 'beta', 'poseCoeff', 'err', 'trans', 'KPS3D', 'fullpose']
                                        hand_info = loadPickleData(pkl_pth)

                                        #generate point cloud
                                        depth3d = o3d.geometry.Image(depth2d * large_hand_mask[:, :, 0])
                                        # load point cloud from depth, NOTE: If scene reconstruction is here, it will be better.
                                        intrinsics = o3d.camera.PinholeCameraIntrinsic(
                                            1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])  # 不用/1000?
                                        pcd = o3d.geometry.PointCloud.create_from_depth_image(
                                            depth3d, intrinsics, stride=2)  # 5693 points
                                        voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # 1345 points

                                        pcd = np.asarray(pcd.points)
                                        voxel_down_pcd = np.asarray(voxel_down_pcd.points)
                                        random_ids = np.random.randint(0, voxel_down_pcd.shape[0], 778)
                                        voxel_down_pcd = voxel_down_pcd[random_ids]  # 778 points

                                        batch_size = 1
                                        ncomps = 30
                                        beta_fixed = np.array([-1.7222083,   0.23059624,  0.4204884, - 4.190077,   0.5057526, - 3.8649838,
                                                            - 2.1274824,   0.8049928,   1.48823,    3.9970665, ])
                                        random_pose = np.random.randn(ncomps+3)
                                        random_trans = np.array([0, 0, 0.5])

                                        cuda_device = FLAGS.gpu

                                        print('crop:', crop_list)
                                        crop_list_renderer = [x1 - int(np.round(camMat[1, 2])) + 960, x2 - int(np.round(camMat[1, 2])) + 960,
                                                            y1 - int(np.round(camMat[0, 2])) + 960, y2 - int(np.round(camMat[0, 2])) + 960]

                                        model = HandObj(batch_size, ncomps, hand_info['poseCoeff'], hand_info['trans'], hand_info['beta'], cur_kps2D, cur_vis,
                                                        large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, crop_list_renderer, flat_bool=False)

                                        model.to(cuda_device)

                                        pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wd_loss, results = model()

                                        result2D_init = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
                                        kpsimg = showHandJoints(color, result2D_init, filename=os.path.join(
                                            outdir, 'p12_rendered_kps_init.png'))

                                        results_seg = results['seg'].detach().cpu().numpy()
                                        rgb_image = cv2.imread(color2d_pth)
                                        rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array(
                                            (255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
                                        cv2.imwrite(os.path.join(
                                            outdir, 'p4_init_rendered_seg.png'), rgb_image)

                                        o3d.io.write_point_cloud(os.path.join(
                                            outdir, './init_pc.ply'), results['pointcloud'])

                                        L_pcd = []
                                        L_seg = []
                                        L_dpt = []
                                        L_kps = []
                                        L_cons = []
                                        optimizer = optim.Adam(model.parameters(), lr=0.01)  # 0.001
                                        torch.autograd.set_detect_anomaly(True)

                                        for _ in tqdm(range(1000)):
                                            pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wrist_depth_loss, _ = model()

                                            loss = pcd_loss * 1 + seg_loss * 1e3 + dpt_loss * 1e3 + \
                                                kps_loss * 1 + cmin_loss * 5e1 + cmax_loss * 5e1 + \
                                                inv_loss * 10e1  # + wrist_depth_loss * 1e4
                                            optimizer.zero_grad()
                                            loss.backward()
                                            L_pcd.append(pcd_loss.item())
                                            L_seg.append(seg_loss.item())
                                            L_dpt.append(dpt_loss.item())
                                            L_kps.append(kps_loss.item())
                                            L_cons.append(cmin_loss.item() + cmax_loss.item() + inv_loss.item())
                                            optimizer.step()

                                        _, _, _, _, _, _, _, _, results = model()
                                        theta = model.theta.data.detach().cpu().numpy()  # [[]]
                                        beta = model.beta.data.detach().cpu().numpy()  # [[]]
                                        trans = model.trans.data.detach().cpu().numpy()  # []
                                        newDict = {
                                            'poseCoeff': theta,
                                            'beta': beta,
                                            'trans': trans
                                        }

                                        with open(os.path.join(outdir, 'newhand.pickle'), 'wb') as f:
                                            pickle.dump(newDict, f)

                                        plt.plot(L_pcd[:])
                                        plt.savefig(os.path.join(outdir, 'p6_L_pcd.png'))
                                        plt.cla()

                                        plt.plot(L_seg[:])
                                        plt.savefig(os.path.join(outdir, 'p7_L_seg.png'))
                                        plt.cla()

                                        plt.plot(L_dpt[:])
                                        plt.savefig(os.path.join(outdir, 'p8_L_dpt.png'))
                                        plt.cla()

                                        plt.plot(L_kps[:])
                                        plt.savefig(os.path.join(outdir, 'p9_L_kps.png'))
                                        plt.cla()

                                        plt.plot(L_cons[:])
                                        plt.savefig(os.path.join(outdir, 'p10_L_cons.png'))
                                        plt.cla()

                                        results_seg = results['seg'].detach().cpu().numpy()
                                        rgb_image = cv2.imread(color2d_pth)
                                        rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array(
                                            (255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
                                        cv2.imwrite(os.path.join(
                                            outdir, 'p1_rendered_seg.png'), rgb_image)

                                        result2D = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
                                        # print(result2D)
                                        o3d.io.write_point_cloud(os.path.join(
                                            outdir, './opt_pc.ply'), results['pointcloud'])

                                        kpsimg = showHandJoints(color, result2D, filename=os.path.join(
                                            outdir, 'p11_rendered_kps.png'))
                                        for kps_coord in kps_anno['coord']:
                                            cv2.circle(kpsimg, center=(int(kps_coord[0]), int(
                                                kps_coord[1])), radius=3, color=[255, 255, 255], thickness=-1)
                                        cv2.imwrite(os.path.join(outdir, 'p12_rendered_kps+gt.png'), kpsimg)

    


if __name__ == '__main__':
        run()
