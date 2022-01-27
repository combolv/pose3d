from pkgutil import extend_path
import cv2
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from loadfile import *
from vis import *
from random import shuffle
from model import HandObj
from tqdm import tqdm
import argparse
from os.path import join
import random
import datetime
import time
import multiprocessing as mlp

random.seed()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')
parser.add_argument('--id', type=str, default='299', help='image id')
FLAGS = parser.parse_args()

jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

DATA_PTH = '/mnt/8T/HOI4D_data_yiqi'
INIT_PTH = '/mnt/8T/HOI4D_data_all/C1C5'
OUT_PTH = '/mnt/8T/HOI4D_data_all/handpose_10'

def getFramewisePose(fileList, outdir):
    camparam_pth = INIT_PTH+'/camera_params/'+outdir.split('/')[-7]+'/color_intrin.npy'
    camMat = np.load(camparam_pth)

    preframe = None
    cuda_device = FLAGS.gpu

    print(outdir)

    for file in fileList:
        print(datetime.datetime.now(), 'opting ', file)
        seq = '/'.join(file.split('/')[:-1])
        image_id = file.split('/')[-1]
        logdir = join(outdir,image_id)
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
            if(kps_anno['label'][kps_i] < 21):
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

        # large_obj_mask, large_hand_mask = read_mask_reverse_H2O(mask_pth)
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
        if pcd.shape[0] == 0 :
            print(file, 'mask is invalid !')
            continue
        voxel_down_pcd = np.asarray(voxel_down_pcd.points)
        random_ids = np.random.randint(0, voxel_down_pcd.shape[0], 778)
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
                    model.parameters(), lr=0.03)  # 0.001
                torch.autograd.set_detect_anomaly(True)

                for _ in range(300):
                    pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wrist_depth_loss, _ = model()

                    loss = pcd_loss * 1 + seg_loss * 2e3 + dpt_loss * 1e3 + \
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
            
            pose_init = np.array([1.3467209,  1.7653652, -1.207203, -0.7234913, -0.46141928,
                                  0.09752903,  0.23736428, -1.4941185, -0.68343824,  0.8829296,
                                  -0.714082, -2.4715464, -1.6300141, -0.22078326,  1.8030486,
                                  -0.96339303,  1.7424369,  1.2276102, -0.443279,  4.624876,
                                  -0.83144915,  1.0743333,  0.2080949, -3.347285,  1.2317966,
                                  0.9432739, -3.7041862,  1.5157006,  0.36411017, -0.6229869,
                                  -0.74925786,  3.17696,  3.5830069])
            # beta_init = np.array([0.51218486,  0.4664125, -2.956826, -4.1959515, -0.44382223,
            #                        3.3613584, -4.1481476, -8.243719,  4.1283875, -0.29030925])
            beta_init = np.array([0.,  0., 0., 0., 0.,
                                  0., 0., 0.,  0., 0.])
            trans_init = np.array([-0.02789204,  0.13406894,  0.57633495])
            with torch.cuda.device(cuda_device):
                model = HandObj(batch_size, ncomps, pose_init, trans_init, beta_init, cur_kps2D, cur_vis,
                                large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, flat_bool=False)

                model.to(cuda_device)

                L_pcd = []
                L_seg = []
                L_dpt = []
                L_kps = []
                L_cons = []
                optimizer = optim.Adam(model.parameters(), lr=0.01)  # 0.001
                torch.autograd.set_detect_anomaly(True)

                for _ in range(800):
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

def getdataList():
    procdirList = []
    outdirList = []
    cams = os.listdir(INIT_PTH)
    cams = [cam for cam in cams if 'ZY' in cam]
    shuffle(cams)
    for cam in cams:
        camdir = join(DATA_PTH, cam)
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
                objs = os.listdir(catdir)
                shuffle(objs)
                for obj in objs:
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
                                if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene, setting, task)):
                                    os.mkdir(join(OUT_PTH, cam, staff, cat, obj, scene, setting, task))

                                if (cat == 'C2' and task == 'T1' and len(outdirList)<=9):
                                # if True:
                                    outdir = join(OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirList.append(taskdir)
                                    outdirList.append(outdir)
    return procdirList, outdirList


def getdataListT1():
    procdirListC1 = []
    outdirListC1 = []
    procdirListC2 = []
    outdirListC2 = []
    procdirListC3 = []
    outdirListC3 = []
    procdirListC4 = []
    outdirListC4 = []
    procdirListC5 = []
    outdirListC5 = []
    procdirListC6 = []
    outdirListC6 = []
    procdirListC7 = []
    outdirListC7 = []
    procdirListC8 = []
    outdirListC8 = []
    procdirListC9 = []
    outdirListC9 = []
    cams = os.listdir(INIT_PTH)
    cams = [cam for cam in cams if 'ZY' in cam]
    # shuffle(cams)
    for cam in cams:
        camdir = join(DATA_PTH, cam)
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
                objs = os.listdir(catdir)
                # shuffle(objs)
                for obj in objs:
                    objdir = join(catdir, obj)
                    if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj)):
                        os.mkdir(join(OUT_PTH, cam, staff, cat, obj))
                    for scene in os.listdir(objdir):
                        scenedir = join(objdir, scene)
                        if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene)):
                            os.mkdir(
                                join(OUT_PTH, cam, staff, cat, obj, scene))
                        for setting in os.listdir(scenedir):
                            settingdir = join(scenedir, setting)
                            if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene, setting)):
                                os.mkdir(join(OUT_PTH, cam, staff,
                                         cat, obj, scene, setting))
                            for task in os.listdir(settingdir):
                                if not os.path.exists(join(OUT_PTH, cam, staff, cat, obj, scene, setting, task)):
                                    os.mkdir(join(OUT_PTH, cam, staff,
                                             cat, obj, scene, setting, task))
                                if (cat == 'C1' and task == 'T1' and len(outdirListC1) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC1.append(taskdir)
                                    outdirListC1.append(outdir)
                                if (cat == 'C2' and task == 'T1' and len(outdirListC2) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC2.append(taskdir)
                                    outdirListC2.append(outdir)
                                if (cat == 'C3' and task == 'T1' and len(outdirListC3) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC3.append(taskdir)
                                    outdirListC3.append(outdir)
                                if (cat == 'C4' and task == 'T1' and len(outdirListC4) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC4.append(taskdir)
                                    outdirListC4.append(outdir)
                                if (cat == 'C5' and task == 'T1' and len(outdirListC5) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC5.append(taskdir)
                                    outdirListC5.append(outdir)
                                if (cat == 'C6' and task == 'T1' and len(outdirListC6) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC6.append(taskdir)
                                    outdirListC6.append(outdir)
                                if (cat == 'C7' and task == 'T1' and len(outdirListC7) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC7.append(taskdir)
                                    outdirListC7.append(outdir)
                                if (cat == 'C8' and task == 'T1' and len(outdirListC8) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC8.append(taskdir)
                                    outdirListC8.append(outdir)
                                if (cat == 'C9' and task == 'T1' and len(outdirListC9) <= 9):
                                    # if True:
                                    outdir = join(
                                        OUT_PTH, cam, staff, cat, obj, scene, setting, task)
                                    taskdir = join(settingdir, task)
                                    procdirListC9.append(taskdir)
                                    outdirListC9.append(outdir)
        procdirList = procdirListC3+procdirListC4+procdirListC5+procdirListC6+procdirListC7+procdirListC8+procdirListC9
        outdirList = outdirListC3+outdirListC4+outdirListC5+outdirListC6+outdirListC7+outdirListC8+outdirListC9
    return procdirList, outdirList



def getHandPoses(dummy, procdirList, outdirList):
    numClips = len(procdirList)
    for i in range(numClips):

        fileListIn = os.listdir(join(procdirList[i], 'handpose'))
        fileListIn = [join(procdirList[i], f[:-5])
                      for f in fileListIn if 'json' in f]
        # fileListIn = sorted(fileListIn)  
        fileListIn = sorted(fileListIn, key=lambda f: f.split(
            '/')[-1].zfill(5), reverse=False)

        getFramewisePose(fileListIn, outdirList[i])





if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    time_start = time.time()
    procdirList, outdirList = getdataListT1()
    print(outdirList)
    exit(0)
    numThreads = 5
    numdirs = len(procdirList)
    assert len(procdirList) == len(outdirList)

    numDirsPerThread = np.ceil(
        numdirs/numThreads).astype(np.uint32)

    procs = []
    for proc_index in range(numThreads):
        startIdx = proc_index*numDirsPerThread
        endIdx = min(startIdx+numDirsPerThread, numdirs)
        args = ([], procdirList[startIdx:endIdx], outdirList[startIdx:endIdx])
        proc = mlp.Process(target=getHandPoses, args=args)

        proc.start()
        procs.append(proc)

    for i in range(len(procs)):
        procs[i].join()
###

    time_end = time.time()
    print('totally cost', time_end-time_start)
