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
import time
import multiprocessing as mlp


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=2,
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
OUT_PTH = '/mnt/8T/HOI4D_data_all/handpose_tmp'

fileList = ['/mnt/8T/HOI4D_data_yiqi/ZY20210800003/H3/C2/N01/S52/s2/T1/60']
outdir = '/mnt/8T/HOI4D_data_all/handpose_tmp/ZY20210800003/H3/C2/N01/S52/s2/T1'
camparam_pth = INIT_PTH+'/camera_params/' + \
    outdir.split('/')[-7]+'/color_intrin.npy'
camMat = np.load(camparam_pth)

preframe = None
cuda_device = FLAGS.gpu

for file in fileList:
    seq = '/'.join(file.split('/')[:-1])
    image_id = file.split('/')[-1]
    logdir = join(outdir, image_id)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    depth2d_pth = seq+'/align_depth/'+image_id+'.png'
    color2d_pth = seq+'/align_image/'+image_id+'.jpg'
    hand_anno_pth = seq+'/handpose/'+image_id+'.json'
    mask_pth = seq+'/2Dseg/mask/'+image_id.zfill(5)+'.png'

    if os.path.exists(join(outdir, image_id+'.pickle')):
        print(join(outdir, image_id+'.pickle'), ' already exist')
        preframe = file
        continue

        #get anno 2d keypoints and their visibility
    kps_anno = parse_json(hand_anno_pth)
    if kps_anno['num'] == 0:
        print(hand_anno_pth, ' have no hand')
        continue

    cur_kps2D = np.zeros((21, 2))
    cur_vis = np.zeros((21, 1))
    for kps_i in range(kps_anno['num']):
        assert kps_i < 21, 'too many kps!!'
        if(kps_anno['label'][kps_i]<21):
            idx = jointsMap.index(
                kps_anno['label'][kps_i])
            cur_kps2D[idx] = kps_anno['coord'][kps_i]
            if kps_anno['vis'][kps_i] == True:
                cur_vis[idx] = 1.0
            else:
                cur_vis[idx] = 0.8
    cur_vis[0] = 5.0  # !!!

    color = cv2.imread(color2d_pth)
    depth2d = cv2.imread(depth2d_pth, cv2.IMREAD_UNCHANGED)
    depth2d = np.array(depth2d, dtype=np.float32) / 1000  # mm
    large_obj_mask, large_hand_mask = read_mask_crop_wrist(mask_pth,kps_anno)

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
    random_ids = np.random.randint(
        0, voxel_down_pcd.shape[0], 778)
    voxel_down_pcd = voxel_down_pcd[random_ids]  # 778 points

    batch_size = 1
    ncomps = 30

    if preframe is not None:
        preseq = '/'.join(preframe.split('/')[-8:-1])
        preimgID = preframe.split('/')[-1]
        print('init frame '+image_id+' with '+preimgID)
        pkl_pth = join(outdir, preimgID+'.pickle')
        hand_info = loadPickleData(pkl_pth)
        with torch.cuda.device(cuda_device):
            model = HandObj(batch_size, ncomps, hand_info['poseCoeff'], hand_info['trans'], hand_info['beta'], cur_kps2D, cur_vis,
                            large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, flat_bool=False)

            model.to(cuda_device)

            L_pcd = []
            L_seg = []
            L_dpt = []
            L_kps = []
            L_cons = []
            optimizer = optim.Adam(
                model.parameters(), lr=0.02)  # 0.001
            torch.autograd.set_detect_anomaly(True)

            for _ in tqdm(range(500)):
                pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wrist_depth_loss, _ = model()

                loss = pcd_loss * 1 + seg_loss * 1e3 + dpt_loss * 1e3 + \
                    kps_loss * 1 + cmin_loss * 5e2 + cmax_loss * 5e2 + \
                    inv_loss * 1e3  # + wrist_depth_loss * 1e4
                optimizer.zero_grad()
                loss.backward()
                L_pcd.append(pcd_loss.item())
                L_seg.append(seg_loss.item())
                L_dpt.append(dpt_loss.item())
                L_kps.append(kps_loss.item())
                L_cons.append(
                    cmin_loss.item() + cmax_loss.item() + inv_loss.item())
                optimizer.step()

            _, _, _, _, _, _, _, _, results = model()
            theta = model.theta.data.detach().cpu().numpy()  # [[]]
            beta = model.beta.data.detach().cpu().numpy()  # [[]]
            trans = model.trans.data.detach().cpu().numpy()  # []
            newDict = {
                'poseCoeff': theta.squeeze(),
                'beta': beta.squeeze(),
                'trans': trans.squeeze()
            }

            with open(os.path.join(outdir, image_id+'.pickle'), 'wb') as f:
                pickle.dump(newDict, f)

            plt.plot(L_pcd[:])
            plt.savefig(os.path.join(logdir, 'p4_L_pcd.png'))
            plt.cla()

            plt.plot(L_seg[:])
            plt.savefig(os.path.join(logdir, 'p5_L_seg.png'))
            plt.cla()

            plt.plot(L_dpt[:])
            plt.savefig(os.path.join(logdir, 'p6_L_dpt.png'))
            plt.cla()

            plt.plot(L_kps[:])
            plt.savefig(os.path.join(logdir, 'p7_L_kps.png'))
            plt.cla()

            plt.plot(L_cons[:])
            plt.savefig(os.path.join(logdir, 'p8_L_cons.png'))
            plt.cla()

            results_seg = results['seg'].detach().cpu().numpy()
            rgb_image = cv2.imread(color2d_pth)
            rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array(
                (255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
            cv2.imwrite(os.path.join(
                logdir, 'p3_rendered_seg.png'), rgb_image)

            result2D = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
            o3d.io.write_point_cloud(os.path.join(
                logdir, './opt_pc.ply'), results['pointcloud'])

            kpsimg = showHandJoints(color, result2D, filename=os.path.join(
                logdir, 'p2_rendered_kps.png'))
            for kps_coord in kps_anno['coord']:
                cv2.circle(kpsimg, center=(int(kps_coord[0]), int(
                    kps_coord[1])), radius=3, color=[255, 255, 255], thickness=-1)
            cv2.imwrite(os.path.join(logdir, 'p1_rendered_kps+gt.png'), kpsimg)

    else:

        pose_init = np.array([0.58280796,  2.0320692, -0.8994741, -0.5411268, -0.3878802 ,
                                0.1980745,  0.0454903, -1.9732434 , -0.02610829,  1.0749623 ,
                                -1.1019206, -1.6209276, -1.0451345 , -0.18634914,  1.6279253 ,
                                -1.1468018,  1.1666641,  0.7588373 , -1.1029167 ,  5.0315523 ,
                                -0.93605477,  0.6246792, -0.11218645, -3.2948873,  1.1618654 ,
                                0.21769527, -4.1251945,  0.06036749,  0.0218429, -0.66927737,
                                -1.4104608,  2.5715125,  3.4667623 ])
        beta_init = np.array([0.51218486,  0.4664125, -2.956826, -4.1959515, -0.44382223,
                                3.3613584, -4.1481476, -8.243719,  4.1283875, -0.29030925])
        trans_init = np.array([0.16644168, 0.27220228, 0.6006424])
        with torch.cuda.device(cuda_device):
            model = HandObj(batch_size, ncomps, pose_init, trans_init, beta_init, cur_kps2D, cur_vis,
                            large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, flat_bool=False)

            model.to(cuda_device)

            L_pcd = []
            L_seg = []
            L_dpt = []
            L_kps = []
            L_cons = []
            optimizer = optim.Adam(
                model.parameters(), lr=0.01)  # 0.001
            torch.autograd.set_detect_anomaly(True)

            for _ in tqdm(range(800)):
                pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wrist_depth_loss, _ = model()

                loss = pcd_loss * 1 + seg_loss * 1e3 + dpt_loss * 1e3 + \
                    kps_loss * 1 + cmin_loss * 5e2 + cmax_loss * 5e2 + \
                    inv_loss * 1e3  # + wrist_depth_loss * 1e4
                optimizer.zero_grad()
                loss.backward()
                L_pcd.append(pcd_loss.item())
                L_seg.append(seg_loss.item())
                L_dpt.append(dpt_loss.item())
                L_kps.append(kps_loss.item())
                L_cons.append(
                    cmin_loss.item() + cmax_loss.item() + inv_loss.item())
                optimizer.step()

            _, _, _, _, _, _, _, _, results = model()
            theta = model.theta.data.detach().cpu().numpy()  # [[]]
            beta = model.beta.data.detach().cpu().numpy()  # [[]]
            trans = model.trans.data.detach().cpu().numpy()  # []
            newDict = {
                'poseCoeff': theta.squeeze(),
                'beta': beta.squeeze(),
                'trans': trans.squeeze()
            }

            with open(os.path.join(outdir, image_id+'.pickle'), 'wb') as f:
                pickle.dump(newDict, f)

            plt.plot(L_pcd[:])
            plt.savefig(os.path.join(logdir, 'p4_L_pcd.png'))
            plt.cla()

            plt.plot(L_seg[:])
            plt.savefig(os.path.join(logdir, 'p5_L_seg.png'))
            plt.cla()

            plt.plot(L_dpt[:])
            plt.savefig(os.path.join(logdir, 'p6_L_dpt.png'))
            plt.cla()

            plt.plot(L_kps[:])
            plt.savefig(os.path.join(logdir, 'p7_L_kps.png'))
            plt.cla()

            plt.plot(L_cons[:])
            plt.savefig(os.path.join(logdir, 'p8_L_cons.png'))
            plt.cla()

            results_seg = results['seg'].detach().cpu().numpy()
            rgb_image = cv2.imread(color2d_pth)
            rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array(
                (255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
            cv2.imwrite(os.path.join(
                logdir, 'p3_rendered_seg.png'), rgb_image)

            result2D = results['2Djoints'].detach(
            ).cpu().numpy().astype(np.int32)
            o3d.io.write_point_cloud(os.path.join(
                logdir, './opt_pc.ply'), results['pointcloud'])

            kpsimg = showHandJoints(color, result2D, filename=os.path.join(
                logdir, 'p2_rendered_kps.png'))
            for kps_coord in kps_anno['coord']:
                cv2.circle(kpsimg, center=(int(kps_coord[0]), int(
                    kps_coord[1])), radius=3, color=[255, 255, 255], thickness=-1)
            cv2.imwrite(os.path.join(logdir, 'p1_rendered_kps+gt.png'), kpsimg)
    preframe = file
