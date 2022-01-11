import cv2
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from loadfile import *
from vis import *
from model import HandObj
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='cuda:0', help='GPU to use [default: GPU 0]')
FLAGS = parser.parse_args()

DATA_PTH = '/home/jiangche/HOI4D/T1'
jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

if __name__ == '__main__':

    depth2d_pth = DATA_PTH+'/align_depth/299.png'
    color2d_pth = DATA_PTH+'/align_image/299.jpg'
    hand_anno_pth = DATA_PTH+'/handpose/299.json'
    mask_pth = DATA_PTH+'/raw_seg_results/299.png'
    pkl_pth = DATA_PTH+'/singleFrameFit/299.pickle'
    camparam_pth = DATA_PTH+'/color_intrin.npy'

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

    x1, x2, y1, y2, c12, c02 = read_mask2bbox_hand(mask_pth)
    crop_list = [x1, x2, y1, y2]  # 1920*1080 square
    # print(crop_list,'croplist')
    obj_mask, hand_mask = read_mask_reverse(mask_pth, dscale=1, crop=crop_list)
    _, large_hand_mask = read_mask_reverse(mask_pth, dscale=1)
    hand_info = loadPickleData(pkl_pth) #['KPS2D', 'imgID', 'conf', 'beta', 'poseCoeff', 'err', 'trans', 'KPS3D', 'fullpose']

    #generate point cloud
    depth3d = o3d.geometry.Image(depth2d * large_hand_mask[:, :, 0])
    # load point cloud from depth, NOTE: If scene reconstruction is here, it will be better.
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d, intrinsics, stride=2)
    # o3d.visualization.draw_geometries([pcd])
    pcd = np.asarray(pcd.points)


    #crop depth 2d
    depth2d_pad = np.pad(depth2d, ((540, 540), (960, 960)))
    depth2d = depth2d_pad[x1+540:x2+540, y1+960:y2+960]

    batch_size = 1
    ncomps = 30
    beta_fixed = np.array([-1.7222083,   0.23059624,  0.4204884, - 4.190077,   0.5057526, - 3.8649838,
                           - 2.1274824,   0.8049928,   1.48823,    3.9970665, ])
    random_pose = np.random.randn(ncomps+3)
    random_trans = np.zeros(3)

    cuda_device = FLAGS.gpu

    model = HandObj(batch_size, ncomps, hand_info['poseCoeff'], hand_info['trans'], beta_fixed, cur_kps2D, cur_vis,
                    hand_mask, obj_mask, depth2d, pcd, camMat, cuda_device, crop_list)
    model.to(cuda_device)

    pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, results = model()

    result2D_init = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
    kpsimg = showHandJoints(color, result2D_init, filename='p12_rendered_kps_init.png')
    cv2.imwrite('p4_init_rendered_seg.png',
                ((200 * results['seg']).detach().cpu().numpy()).astype(np.uint8))

    cv2.imwrite('p5_init_rendered_depth.png', (1600 * (results['dep'] -
                0.85).detach().cpu().numpy()).astype(np.uint8))

    L_pcd = []
    L_seg = []
    L_dpt = []
    L_kps = []
    L_cons = []
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in tqdm(range(1000)):
        pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, _ = model()
        loss = pcd_loss * 0.01 + seg_loss * 10.0 + dpt_loss * 2.0 + \
            kps_loss * 10 + cmin_loss * 5e2 + cmax_loss * 5e2 + inv_loss * 1e3
        # loss = 0.01 10 2 0 5e2 5e2 1e3
        optimizer.zero_grad()
        loss.backward()
        # print(pcd_loss.item(), seg_loss.item(), dpt_loss.item())
        L_pcd.append(pcd_loss.item())
        L_seg.append(seg_loss.item())
        L_dpt.append(dpt_loss.item())
        L_kps.append(kps_loss.item())
        L_cons.append(cmin_loss.item() + cmax_loss.item() + inv_loss.item())
        # print(model.theta[0:3].data)
        # print(model.theta[0:3].grad)
        # input()
        optimizer.step()

    _, _, _, _, _, _, _, results = model()
    theta = model.theta.data.detach().cpu().numpy()  # [[]]
    beta = model.beta.data.detach().cpu().numpy()  # [[]]
    trans = model.trans.data.detach().cpu().numpy()  # []
    newDict = {
        'poseCoeff': theta,
        'beta': beta,
        'trans': trans
    }

    with open(os.path.join('newhand.pickle'), 'wb') as f:
        pickle.dump(newDict, f)

    plt.plot(L_pcd)
    plt.savefig('p6_L_pcd.png')
    plt.cla()

    plt.plot(L_seg)
    plt.savefig('p7_L_seg.png')
    plt.cla()

    plt.plot(L_dpt)
    plt.savefig('p8_L_dpt.png')
    plt.cla()

    plt.plot(L_kps)
    plt.savefig('p9_L_kps.png')
    plt.cla()

    plt.plot(L_cons)
    plt.savefig('p10_L_cons.png')

    cv2.imwrite('p1_rendered_seg.png', ((
        200 * results['seg']).detach().cpu().numpy()).astype(np.uint8))

    cv2.imwrite('p2_gt_depth.png', (1600*(depth2d-0.85)).astype(np.uint8))

    cv2.imwrite('p3_rendered_depth.png', (1600 *
                (results['dep']-0.85).detach().cpu().numpy()).astype(np.uint8))

    result2D = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
    # print(result2D)

    kpsimg = showHandJoints(color, result2D, filename='p11_rendered_kps.png')
    for kps_coord in kps_anno['coord']:
        cv2.circle(kpsimg, center=(int(kps_coord[0]), int(
            kps_coord[1])), radius=3, color=[255, 255, 255], thickness=-1)
    cv2.imwrite('p12_rendered_kps+gt.png', kpsimg)
