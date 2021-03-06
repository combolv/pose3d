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
parser.add_argument('--id', type=str, default='299', help='image id')
FLAGS = parser.parse_args()

DATA_PTH = '/home/jiangche/HOI4D/T1_old'
jointsMap = [0,
             13, 14, 15, 16,
             1, 2, 3, 17,
             4, 5, 6, 18,
             10, 11, 12, 19,
             7, 8, 9, 20]

if __name__ == '__main__':

    image_id = FLAGS.id

    depth2d_pth = DATA_PTH+'/align_depth/'+image_id+'.png'
    color2d_pth = DATA_PTH+'/align_image/'+image_id+'.jpg'
    hand_anno_pth = DATA_PTH+'/handpose/'+image_id+'.json'
    mask_pth = DATA_PTH+'/raw_seg_results/'+image_id+'.png'
    pkl_pth = DATA_PTH+'/singleFrameFit/'+image_id+'.pickle'
    camparam_pth = DATA_PTH+'/color_intrin.npy'

    OUTPUT_PTH = '/home/jiangche/HOI4D/output'+image_id
    if not os.path.exists(OUTPUT_PTH):
        os.mkdir(OUTPUT_PTH)


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
    cur_vis[0] = 5.0#!!!

    x1, x2, y1, y2 = read_mask2bbox_hand(mask_pth)
    crop_list = [x1, x2, y1, y2]  # 1920*1080 square
    # print(crop_list,'croplist')
    obj_mask, hand_mask = read_mask_reverse(mask_pth, dscale=1, crop=crop_list)
    large_obj_mask, large_hand_mask = read_mask_reverse(mask_pth, dscale=1)
    hand_info = loadPickleData(pkl_pth) #['KPS2D', 'imgID', 'conf', 'beta', 'poseCoeff', 'err', 'trans', 'KPS3D', 'fullpose']

    #generate point cloud
    depth3d = o3d.geometry.Image(depth2d * large_hand_mask[:, :, 0])
    # load point cloud from depth, NOTE: If scene reconstruction is here, it will be better.
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2])  # ??????/1000?
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d, intrinsics, stride=2)  # 5693 points
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.005)  # 1345 points
    # o3d.visualization.draw_geometries([pcd])
    # o3d.io.write_point_cloud('./pcd.ply', pcd)
    # o3d.io.write_point_cloud('./voxel_down_pcd.ply', voxel_down_pcd)
    pcd = np.asarray(pcd.points)
    voxel_down_pcd = np.asarray(voxel_down_pcd.points)
    random_ids = np.random.randint(0, voxel_down_pcd.shape[0], 778)
    voxel_down_pcd = voxel_down_pcd[random_ids] # 778 points

    '''
    # ??????: ??????????????????(3D??????)?????????????????????(????????????)
    ans = pcd.transpose()
    ans = np.dot(camMat, ans)
    ans = ans.transpose()
    ans1 = (ans[:, 0] / ans[:, 2]).reshape(-1, 1)
    ans2 = (ans[:, 1] / ans[:, 2]).reshape(-1, 1)
    print(np.concatenate((ans1, ans2), axis=1))
    '''

    #crop depth 2d
    # depth2d = depth2d[x1:x2,y1:y2]

    batch_size = 1
    ncomps = 30
    pose_init = np.array([1.3467209,  1.7653652, -1.207203, -0.7234913, -0.46141928,
                          0.09752903,  0.23736428, -1.4941185, -0.68343824,  0.8829296,
                          -0.714082, -2.4715464, -1.6300141, -0.22078326,  1.8030486,
                          -0.96339303,  1.7424369,  1.2276102, -0.443279,  4.624876,
                          -0.83144915,  1.0743333,  0.2080949, -3.347285,  1.2317966,
                          0.9432739, -3.7041862,  1.5157006,  0.36411017, -0.6229869,
                          -0.74925786,  3.17696,  3.5830069])
    # beta_init = np.array([0.92336965,  3.7251637,  2.5664854, -6.3361144,  0.73302877,
    #                       -5.627644,  1.6466948, -0.7867594,  4.238981,  0.24937129])
    beta_init = np.zeros(10)
    trans_init = np.array([-0.02789204,  0.13406894,  0.57633495])

    cuda_device = FLAGS.gpu

    # model = HandObj(batch_size, ncomps, hand_info['poseCoeff'], hand_info['trans'], beta_fixed, cur_kps2D, cur_vis,
    #                 hand_mask, obj_mask, depth2d, pcd, camMat, cuda_device, crop_list)
    print('crop:',crop_list)
    crop_list_renderer = [x1 - int(np.round(camMat[1, 2])) + 960, x2 - int(np.round(camMat[1, 2])) + 960,
                 y1 - int(np.round(camMat[0, 2])) + 960, y2 - int(np.round(camMat[0, 2])) + 960]

    # model = HandObj(batch_size, ncomps, hand_info['poseCoeff'], hand_info['trans'], hand_info['beta'], cur_kps2D, cur_vis,
    #                  large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, crop_list_renderer,flat_bool=False)
    model = HandObj(batch_size, ncomps, pose_init, trans_init, beta_init, cur_kps2D, cur_vis,
                     large_hand_mask, large_obj_mask, depth2d, voxel_down_pcd, camMat, cuda_device, crop_list_renderer,flat_bool=False)

    model.to(cuda_device)

    pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss,wd_loss, results = model()

    result2D_init = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
    kpsimg = showHandJoints(color, result2D_init, filename=os.path.join(OUTPUT_PTH, 'p12_rendered_kps_init.png'))

    results_seg = results['seg'].detach().cpu().numpy()
    rgb_image = cv2.imread(color2d_pth)
    rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array((255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_PTH, 'p4_init_rendered_seg.png'), rgb_image)

    o3d.io.write_point_cloud(os.path.join(OUTPUT_PTH, './init_pc.ply'), results['pointcloud'])


    # cv2.imwrite(os.path.join(OUTPUT_PTH,'p5_init_rendered_depth.png'), (1600 * (results['dep'] -
    #             0.85).detach().cpu().numpy()).astype(np.uint8))
    # print(results['dep'].detach().cpu().numpy())
    # vis_model_dpt(depth2d_pth, results['dep'], os.path.join(OUTPUT_PTH,'p5_init_rendered_depth.png'),
    #         crop_list)

    L_pcd = []
    L_seg = []
    L_dpt = []
    L_kps = []
    L_cons = []
    # L_wd = []
    optimizer = optim.Adam(model.parameters(), lr=0.03)  # 0.001
    torch.autograd.set_detect_anomaly(True)


    for _ in tqdm(range(300)):
        pcd_loss, seg_loss, dpt_loss, kps_loss, cmin_loss, cmax_loss, inv_loss, wrist_depth_loss, _ = model()

        loss = pcd_loss * 1 + seg_loss * 1e3 + dpt_loss * 1e3 + \
            kps_loss * 1 + cmin_loss * 5e2 + cmax_loss * 5e2 + inv_loss * 1e3# + wrist_depth_loss * 1e4
        optimizer.zero_grad()
        loss.backward()
        L_pcd.append(pcd_loss.item())
        L_seg.append(seg_loss.item())
        L_dpt.append(dpt_loss.item())
        L_kps.append(kps_loss.item())
        L_cons.append(cmin_loss.item() + cmax_loss.item() + inv_loss.item())
        # L_wd.append(wrist_depth_loss.item())
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

    with open(os.path.join(OUTPUT_PTH, 'newhand.pickle'), 'wb') as f:
        pickle.dump(newDict, f)

    plt.plot(L_pcd[:])
    plt.savefig(os.path.join(OUTPUT_PTH,'p6_L_pcd.png'))
    plt.cla()

    plt.plot(L_seg[:])
    plt.savefig(os.path.join(OUTPUT_PTH, 'p7_L_seg.png'))
    plt.cla()

    plt.plot(L_dpt[:])
    plt.savefig(os.path.join(OUTPUT_PTH,'p8_L_dpt.png'))
    plt.cla()

    plt.plot(L_kps[:])
    plt.savefig(os.path.join(OUTPUT_PTH,'p9_L_kps.png'))
    plt.cla()

    plt.plot(L_cons[:])
    plt.savefig(os.path.join(OUTPUT_PTH,'p10_L_cons.png'))
    plt.cla()

    # plt.plot(L_wd[:])
    # plt.savefig(os.path.join(OUTPUT_PTH, 'p13_L_wd.png'))
    # plt.cla()


    results_seg = results['seg'].detach().cpu().numpy()
    rgb_image = cv2.imread(color2d_pth)
    rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2] += np.array(
        (255 - rgb_image[:results_seg.shape[0], :results_seg.shape[1], 2]) * results_seg, dtype=np.uint8)
    cv2.imwrite(os.path.join(
        OUTPUT_PTH, 'p1_rendered_seg.png'), rgb_image)

    # vis_depth(depth2d*1000, os.path.join(OUTPUT_PTH, 'p2_gt_depth.png'))

    # vis_model_dpt(depth2d_pth, results['dep'], os.path.join(OUTPUT_PTH, 'p3_rendered_depth.png'),
    #               crop_list)


    result2D = results['2Djoints'].detach().cpu().numpy().astype(np.int32)
    # print(result2D)
    o3d.io.write_point_cloud(os.path.join(OUTPUT_PTH, './opt_pc.ply'), results['pointcloud'])

    kpsimg = showHandJoints(color, result2D, filename=os.path.join(OUTPUT_PTH, 'p11_rendered_kps.png'))
    for kps_coord in kps_anno['coord']:
        cv2.circle(kpsimg, center=(int(kps_coord[0]), int(
            kps_coord[1])), radius=3, color=[255, 255, 255], thickness=-1)
    cv2.imwrite(os.path.join(OUTPUT_PTH, 'p12_rendered_kps+gt.png'), kpsimg)
