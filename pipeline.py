import torch
# from tqdm import tqdm
from torch import optim
import time
# import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rt
from loadfile import annoed_path_generator_from_total_json, every_path_generator_from_total_json

from loadfile import path_list2plainobj_input
from interp import get_all_poses_from_0json_path_and_output_log_path
from interp import get_large_gap_poses_from_0json_path_and_output_log_path
# from model import PlainObj
from tqdm import tqdm
from vis import vis2d, vis3d, vis_depth, vis_model_dpt

class ManualBar:
    def __init__(self, max_iter):
        self.max_iter = max_iter
        self.now_iter = 0
        self.total_time = 0

    def start(self):
        self.start_time = time.time()

    def end(self):
        def f(sec):
            return str(int(sec // 60)).zfill(2) + ':' + str(int(sec % 60)).zfill(2)

        self.now_iter += 1
        now_gap = time.time() - self.start_time
        self.total_time += now_gap
        avg_time = self.total_time / self.now_iter
        if now_gap > avg_time * 2:
            avg_time = now_gap
        rest_iter = self.max_iter - self.now_iter
        est_time = avg_time * rest_iter

        print(f(now_gap), '[', self.now_iter, '/', self.max_iter, ']',
              '[', f(self.total_time), '<', f(est_time), ']')

    def tick(self):
        self.end()
        self.start()


def draw_histo(x, title, xmin=None, xmax=None, save_proto=True):
    import matplotlib.pyplot as plt
    n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.title(title)
    if xmin is not None:
        plt.xlim(xmin=xmin, xmax=xmax)
    plt.show()
    if save_proto:
        np.save('./'+title+'.npy', np.array(x))


def draw_doulbe_histo(x, y, title, xmin=None, xmax=None):
    n, bins, patches = plt.hist(x=x, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.title(title)
    if xmin is not None:
        plt.xlim(xmin=xmin, xmax=xmax)
    n, bins, patches = plt.hist(x=y, bins=bins, color='#aa0504',
                                alpha=0.7, rwidth=0.85)
    plt.show()


def check_gt_loss_terms_gap5(cuda_device='cuda:0'):
    all_inter_pcd_loss = []
    all_inter_seg_loss = []
    all_inter_dpt_loss = []
    ticker = ManualBar(136)
    ticker.start()
    for path_lists in every_path_generator_from_total_json('total.json', range(0, 299, 5)):
        inter_rot, inter_trans = get_all_poses_from_0json_path_and_output_log_path(path_lists[0][3], path_lists[0][5])
        for i, path_list in enumerate(path_lists):
            model_input = path_list2plainobj_input(*path_list[:5])
            model_input[0] = inter_rot[i * 5]
            model_input[1] = inter_trans[i * 5]
            init_model = PlainObj(*model_input)
            init_model.to(cuda_device)
            with torch.no_grad():
                try:
                    pcd_loss, seg_loss, dpt_loss, _, _ = init_model()
                except RuntimeError:
                    all_inter_pcd_loss.append(-99.)
                    all_inter_seg_loss.append(-.5)
                    all_inter_dpt_loss.append(-.5)
                    continue
                all_inter_pcd_loss.append(pcd_loss.item())
                all_inter_seg_loss.append(seg_loss.item())
                all_inter_dpt_loss.append(dpt_loss.item())
        ticker.tick()

    draw_histo(all_inter_pcd_loss, 'all_inter_pcd_loss_exp1_2')
    draw_histo(all_inter_seg_loss, 'all_inter_seg_loss_exp1_2')
    draw_histo(all_inter_dpt_loss, 'all_inter_dpt_loss_exp1_2')


def check_gt_loss_terms_gap10(cuda_device='cuda:0'):
    all_inter_pcd_loss = []
    all_inter_seg_loss = []
    all_inter_dpt_loss = []
    ticker = ManualBar(136)
    ticker.start()
    for path_lists in every_path_generator_from_total_json('total.json', range(10, 299, 20)):
        gt_rot, gt_trans, inter_rot, inter_trans = get_large_gap_poses_from_0json_path_and_output_log_path(path_lists[0][3], path_lists[0][5])
        for i, path_list in enumerate(path_lists):
            # gap_rot = (Rt.from_rotvec(gt_rot[i]) * Rt.from_rotvec(inter_rot[i]).inv()).as_rotvec()
            # if np.linalg.norm(gap_rot) < (5. / 180.) * np.pi:
            #     continue
            model_input = path_list2plainobj_input(*path_list[:5])
            model_input[0] = gt_rot[i]
            model_input[1] = gt_trans[i]
            init_model = PlainObj(*model_input)
            init_model.to(cuda_device)
            with torch.no_grad():
                try:
                    pcd_loss, seg_loss, dpt_loss, seg_gt, dep_gt = init_model()
                except RuntimeError:
                    continue
                all_inter_pcd_loss.append(pcd_loss.item())
                all_inter_seg_loss.append(seg_loss.item())
                all_inter_dpt_loss.append(dpt_loss.item())
            model_input[0] = inter_rot[i]
            model_input[1] = inter_trans[i]
            init_model = PlainObj(*model_input)
            init_model.to(cuda_device)
            with torch.no_grad():
                try:
                    pcd_loss, seg_loss, dpt_loss, seg, dep = init_model()
                except RuntimeError:
                    all_inter_pcd_loss.pop()
                    all_inter_seg_loss.pop()
                    all_inter_dpt_loss.pop()
                    continue
                vis_model_dpt(path_list[1], dep, 'depout.jpg', model_input[-1])
                input()
                # if pcd_loss.item() - all_inter_pcd_loss[-1] < -50:
                #     vis2d(dep, seg, path_list[-1], model_input[-1], 'inter.jpg')
                #     vis2d(dep_gt, seg_gt, path_list[-1], model_input[-1], 'gt.jpg')
                #     vis3d(gt_rot[i], gt_trans[i], path_list[-1], path_list[1], path_list[-3], path_list[0], model_input[-1], path_list[3])
                #     vis3d(inter_rot[i], inter_trans[i], path_list[-1], path_list[1], path_list[-3], path_list[0], model_input[-1], path_list[3])
                #     print(path_list)
                #     input('continue?')
                all_inter_pcd_loss.append(pcd_loss.item())
                all_inter_seg_loss.append(seg_loss.item())
                all_inter_dpt_loss.append(dpt_loss.item())
        ticker.tick()

    draw_histo(all_inter_pcd_loss, 'gap10_correct_depth_pcd_loss_exp1_4')
    draw_histo(all_inter_seg_loss, 'gap10_correct_depth_seg_loss_exp1_4')
    draw_histo(all_inter_dpt_loss, 'gap10_correct_depth_dpt_loss_exp1_4')


def anna_result_check_gt_loss_terms():

    all_inter_pcd_loss = np.load('gap10_correct_depth_pcd_loss_exp1_4.npy')
    all_inter_seg_loss = np.load('gap10_correct_depth_seg_loss_exp1_4.npy')
    all_inter_dpt_loss = np.load('gap10_correct_depth_dpt_loss_exp1_4.npy')

    histo2 = all_inter_dpt_loss[::2][:90]
    histo1 = all_inter_dpt_loss[1::2][:90]
    x = range(len(histo1))
    ticks = [15,30,45,60,75]
    plt.title('depth')
    plt.ylim(ymin=-.05, ymax=.05)
    plt.plot(x, histo1, label='interpolate')
    plt.plot(x, histo2, label='ground truth')
    plt.plot(x, histo1 - histo2, label='delta')
    plt.plot(x, [0]*len(x), label='y=0')
    plt.xticks(ticks)
    plt.legend()
    plt.show()


def get_all_path_list():
    all_list = []
    for path_lists in every_path_generator_from_total_json('total.json'):
        all_list.append(path_lists)
    torch.save(all_list, 'mug_list.pt')


# def ground_detect

def optim_pipeline(cuda_num=0, start_pos=0, object_name="mug", end_pos=100):
    assert object_name in ['mug', 'bottle']
    print('Start:cuda=', cuda_num)
    with torch.cuda.device(cuda_num):
        from model import PlainObj
        out_path = '/home/yunze/pose3d_lkb/'+ object_name + '_check/'
        all_list = torch.load(object_name + '_list.pt')
        for j, path_lists in enumerate(all_list):
            if j < start_pos or j > end_pos:
                continue
            if os.path.exists(out_path + str(j) + '/trans.npy'):
                continue
            # new_out_path = path_lists[0][5].replace('20211110', 'seg_3dnew/seg_3d')
            # if not os.path.exists(new_out_path):
            #     new_out_path = path_lists[0][5]
            inter_rot, inter_trans = get_all_poses_from_0json_path_and_output_log_path(path_lists[0][3], path_lists[0][5])
            input()
            if inter_rot is None:
                continue
            fin_all_rot, fin_all_trans = [], []
            try:
                for i, path_list in tqdm(enumerate(path_lists)):
                    # print(len(inter_rot), len(path_lists))
                    # input()
                    model_input = path_list2plainobj_input(*path_list[:5], inter_rot[i], inter_trans[i])
                    # try:
                    init_model = PlainObj(*model_input)
                    init_model.cuda()
                    pcd_loss, seg_loss, dpt_loss, dep, seg = init_model()
                    a = pcd_loss * 300 + seg_loss
                    a.backward()
                    # except RuntimeError as e:
                    #     print(e)
                    #     break
                    print('Start', j , ":", i)
                    vis2d(dep, seg, path_list[-1], model_input[-1], out_path + str(j) + '/before/' + str(i) + '.png')
                    init_model = PlainObj(*model_input)
                    init_model.cuda()
                    optimizer = optim.Adam(init_model.parameters(), lr=0.001)
                    all_pcd_loss = []
                    all_seg_loss = []
                    all_dpt_loss = []
                    dep, seg = None, None
                    for _ in range(50):
                        optimizer.zero_grad()
                        pcd_loss, seg_loss, dpt_loss, dep, seg = init_model()
                        all_loss = pcd_loss * 100 + seg_loss + 0.2 * dpt_loss # / seg_loss.item()  + dpt_loss / dpt_loss.item()
                        all_loss.backward()
                        optimizer.step()
                        all_pcd_loss.append(100 * pcd_loss.item())
                        all_seg_loss.append(seg_loss.item())
                        all_dpt_loss.append(dpt_loss.item())
                    # final_rot, final_trans = init_model.rotvec.detach().cpu().numpy(), init_model.trans.detach().cpu().numpy()
                    vis2d(dep, seg, path_list[-1], model_input[-1], out_path + str(j) + '/after/' + str(i) + '.png')

                    # x = list(range(50))
                    # print(all_pcd_loss, all_dpt_loss, all_seg_loss)
                    # input()
                    fin_rot = init_model.rotvec.detach().cpu().numpy()
                    fin_trans = init_model.trans.detach().cpu().numpy()
                    # plt.plot(x, all_pcd_loss, label='cd')
                    # plt.plot(x, all_seg_loss, label='seg')
                    # plt.plot(x, all_dpt_loss, label='dep')
                    # plt.legend()
                    if not os.path.exists(out_path + str(j) + '/loss/'):
                        os.mkdir(out_path + str(j) + '/loss/')
                    # plt.savefig(out_path + str(j) + '/loss/' + str(i) + '.jpg')
                    # plt.close('all')
                    # drawall_dpt_loss
                    fin_all_rot.append(fin_rot)
                    fin_all_trans.append(fin_trans)
                else:
                    to_save_rot = np.array(fin_all_rot)
                    to_save_trans = np.array(fin_all_trans)
                    np.save(out_path + str(j) + '/rot.npy', to_save_rot)
                    np.save(out_path + str(j) + '/trans.npy', to_save_trans)
                    print('fin:', j)
            except:
                continue


def optim_check_pipeline(cuda_num=0, start_pos=0, object_name="bottle", end_pos=100):
    assert object_name in ['mug', 'bottle']
    print('Start:cuda=', cuda_num)
    with torch.cuda.device(cuda_num):
        from model import PlainObj
        out_path = '/home/yunze/pose3d_lkb/'+ object_name + '_check/'
        all_list = torch.load(object_name + '_list.pt')
        for j, path_lists in enumerate(all_list):
            if j < start_pos or j > end_pos:
                continue
            if os.path.exists(out_path + str(j) + 'trans.npy'):
                continue

            _, _, inter_rot, inter_trans = get_large_gap_poses_from_0json_path_and_output_log_path(path_lists[0][3], path_lists[0][5])

            if inter_rot is None:
                continue
            fin_all_rot, fin_all_trans = [], []
            try:
                for k, i in enumerate(range(10, 301, 20)):
                    path_list = path_lists[i]
                    model_input = path_list2plainobj_input(*path_list[:5], inter_rot[k], inter_trans[k])
                    # try:
                    init_model = PlainObj(*model_input)
                    init_model.cuda()
                    pcd_loss, seg_loss, dpt_loss, dep, seg = init_model()
                    a = pcd_loss * 300 + seg_loss
                    a.backward()
                    # except RuntimeError as e:
                    #     print(e)
                    #     break
                    print('Start', j , ":", i)
                    vis2d(dep, seg, path_list[-1], model_input[-1], out_path + str(j) + '/before/' + str(i) + '.png')
                    init_model = PlainObj(*model_input)
                    init_model.cuda()
                    optimizer = optim.Adam(init_model.parameters(), lr=0.001)
                    all_pcd_loss = []
                    all_seg_loss = []
                    all_dpt_loss = []
                    dep, seg = None, None
                    for _ in range(50):
                        optimizer.zero_grad()
                        pcd_loss, seg_loss, dpt_loss, dep, seg = init_model()
                        all_loss = pcd_loss * 100 + seg_loss + 0.2 * dpt_loss # / seg_loss.item()  + dpt_loss / dpt_loss.item()
                        all_loss.backward()
                        optimizer.step()
                        all_pcd_loss.append(100 * pcd_loss.item())
                        all_seg_loss.append(seg_loss.item())
                        all_dpt_loss.append(dpt_loss.item())

                    fin_rot = init_model.rotvec.detach().cpu().numpy()
                    fin_trans = init_model.trans.detach().cpu().numpy()

                    fin_all_rot.append(fin_rot)
                    fin_all_trans.append(fin_trans)

                to_save_rot = np.array(fin_all_rot)
                to_save_trans = np.array(fin_all_trans)
                np.save(out_path + str(j) + '/fake_rot.npy', to_save_rot)
                np.save(out_path + str(j) + '/fake_trans.npy', to_save_trans)
                print('fin:', j)
            except:
                print('failed:',j)
                continue


def optim_check(object_name="bottle"):
    def compute_error(rotA, rotB, transA, transB):
        rot_z1, rot_z2 = Rt.from_rotvec(rotA).as_matrix()[..., 2], Rt.from_rotvec(rotB).as_matrix()[..., 2]
        rel_rot_vec = np.sum(rot_z1 * rot_z2, axis=-1)
        angle_degree = np.arccos(rel_rot_vec) * 180 / np.pi
        trans_cm = np.linalg.norm(transA - transB, axis=-1) * 100
        return angle_degree, trans_cm
    out_path = '/home/yunze/pose3d_lkb/' + object_name + '_check/'
    all_list = torch.load(object_name + '_list.pt')
    all_key = ['anno', 'inter', 'refined_anno', 'refined_inter']
    all_rot = {'anno':[], 'inter':[], 'refined_anno':[], 'refined_inter':[]}
    all_trans = {'anno':[], 'inter':[], 'refined_anno':[], 'refined_inter':[]}
    for j, path_lists in enumerate(all_list):
        if not os.path.exists(out_path + str(j) + "/trans.npy"):
            continue
        anno_rot, anno_trans, inter_rot, inter_trans = get_large_gap_poses_from_0json_path_and_output_log_path(path_lists[0][3],
                                                                                               path_lists[0][5])

        refined_rot, refined_trans = np.load(out_path + str(j) + "/rot.npy"), np.load(out_path + str(j) + "/trans.npy")
        refined_fake_rot, refined_fake_trans = np.load(out_path + str(j) + '/fake_rot.npy'), np.load(out_path + str(j) + '/fake_trans.npy')
        for k, i in enumerate(range(10, 301, 20)): # k: 0,1,2,..., i: 10, 30, ...
            all_rot['refined_anno'].append(refined_rot[i])
            all_trans['refined_anno'].append(refined_trans[i])
            all_rot['anno'].append(anno_rot[k])
            all_trans['anno'].append(anno_trans[k])
            all_rot['inter'].append(inter_rot[k])
            all_trans['inter'].append(inter_trans[k])
            all_rot['refined_inter'].append(refined_fake_rot[k])
            all_trans['refined_inter'].append(refined_fake_trans[k])

    for key in all_key:
        all_rot[key] = np.array(all_rot[key])
        all_trans[key] = np.array(all_trans[key])
        print(key, all_rot[key].shape, all_trans[key].shape)

    err_rot = {}
    err_trans = {}
    for i in range(3):
        for j in range(i+1, 4):
            key = all_key[i] + '&' +all_key[j]
            err_rot[key], err_trans[key] = compute_error(
                all_rot[all_key[i]],
                all_rot[all_key[j]],
                all_trans[all_key[i]],
                all_trans[all_key[j]]
            )
    for key in err_rot:
        print(key, ":", err_rot[key].shape)
        print(key, ":", err_trans[key].shape)
        input()

    torch.save(err_rot, 'bottle_err_rot.pt')
    torch.save(err_trans, 'bottle_err_trans.pt')





def load_articulated_test(articulated_path):
    a_path = "/mnt/8T/HOI4D_CAD_Model/mobility_annotations/笔记本电脑/041/"
    import open3d as o3d
    print(a_path + 'objs/new-1.obj')
    assert os.path.exists(a_path + 'objs/new-1.obj')
    assert os.path.exists(a_path + 'objs/new-2.obj')
    new_1 = o3d.io.read_triangle_mesh(a_path + 'objs/new-1.obj')
    voxel_size = max(new_1.get_max_bound() - new_1.get_min_bound()) / 32
    mesh_smp1 = new_1.simplify_vertex_clustering(voxel_size=voxel_size,
                                               contraction=o3d.geometry.SimplificationContraction.Average)

    new_0 = o3d.io.read_triangle_mesh(a_path + 'objs/new-2.obj')
    voxel_size = max(new_0.get_max_bound() - new_0.get_min_bound()) / 32
    mesh_smp0 = new_0.simplify_vertex_clustering(voxel_size=voxel_size,
                                                 contraction=o3d.geometry.SimplificationContraction.Average)

    import json
    with open(a_path + 'mobility_v2.json', "r") as f:
        res = json.load(f)
        axis_meta = res[0]["jointData"]
        axis_info, limit = axis_meta["axis"], axis_meta["limit"]
        orig, direction = np.array(axis_info["origin"], dtype=np.float32), np.array(axis_info["direction"], dtype=np.float32)
        direction /= np.linalg.norm(direction)
        t_max, t_min, no_lim = limit["a"], limit["b"], limit["noLimit"]
        rad_min = - t_min / 180 * np.pi
        rad_max = - t_max / 180 * np.pi
        rot_mat = Rt.from_rotvec(direction * rad_max).as_matrix()
        virt_trans = orig - rot_mat @ orig
        new_0.rotate(rot_mat, center=orig)
    prt_list = [new_0, new_1]
    o3d.visualization.draw_geometries(prt_list)


def path_articulated_object():
    a = []
    for r in annoed_path_generator_from_total_json("total.json", True):
        print(r)
        a.append(r)
    torch.save(a, "./articulated_path.pt")


def load_articulated_object():
    a = torch.load("./articulated_path.pt")
    from loadfile import path_list2artobj_input
    art_input = path_list2artobj_input(*a[0][0][:5])
    from model import ArtObj
    init_model = ArtObj(*art_input)
    init_model.cuda()
    _, _, _, seg, dep = init_model()
    # print(a[-1])
    vis2d(dep[-1], seg[0], a[0][0][-1], art_input[-1], './art.jpg')


def load_articulated_object_paths(paths):
    from loadfile import path_list2artobj_input
    art_input = path_list2artobj_input(*paths[:5])
    from model import ArtObj
    import cv2
    init_model = ArtObj(*art_input)
    init_model.cuda()
    _, _, _, seg, dep = init_model()
    # print(a[-1])
    cv2.imwrite('art2.jpg', seg[0].detach().cpu().numpy() * 200)
    vis2d(dep[-1], seg[0], paths[-1], art_input[-1], './art.jpg')
    vis2d(dep[-1], seg[1], paths[-1], art_input[-1], './art1.jpg')


def verify_quality():
    from loadfile import read_rtd
    from vis import showAnnoBox

    def compute_error(rotA, rotB, transA, transB):
        rot_z1, rot_z2 = Rt.from_rotvec(rotA).as_matrix()[..., 2], Rt.from_rotvec(rotB).as_matrix()[..., 2]
        rel_rot_vec = np.sum(rot_z1 * rot_z2, axis=-1)
        angle_degree = np.arccos(rel_rot_vec) * 180 / np.pi
        trans_cm = np.linalg.norm(transA - transB) * 100
        return angle_degree, trans_cm
    bottle_list = torch.load("bottle_list.pt")
    for i, path_lists in enumerate(bottle_list):
        trans_refined_path = '/home/yunze/pose3d_lkb/bottle_check/' + str(i) + '/trans.npy'
        rot_refined_path = '/home/yunze/pose3d_lkb/bottle_check/' + str(i) + '/rot.npy'
        if not os.path.exists(trans_refined_path):
            print(i, ": not ready")
            continue
        json_0frame = path_lists[0][3]
        json_folder = os.path.dirname(json_0frame)
        json_name = os.path.basename(json_0frame)
        zlen = json_name.count('0')
        rot_refined, trans_refined = np.load(rot_refined_path), np.load(trans_refined_path)

        for j in range(0, 301, 10):
            if j == 300:
                j = 299
            model_input = path_list2plainobj_input(*path_lists[j][:5])
            rot_ref, trans_ref = rot_refined[j], trans_refined[j]
            rot_gt, trans_gt, dim = read_rtd(os.path.join(json_folder, str(j).zfill(zlen) + '.json'))
            rot_err, trans_err = compute_error(rot_gt, rot_ref, trans_ref, trans_gt)
            print('id:', i, j, (rot_err, trans_err))
            if trans_err > 4:
                meta = path_lists[j]

                vis3d(rot_gt, trans_gt, meta[-1], meta[1], meta[4], meta[0], model_input[-1], json0_path=meta[3])
                vis3d(rot_ref, trans_ref, meta[-1], meta[1], meta[4], meta[0], model_input[-1], json0_path=meta[3])
                showAnnoBox([(rot_gt, trans_gt, dim)], meta[-1], meta[0], 'bottle_test_gt.png')
                showAnnoBox([(rot_ref, trans_ref, dim)], meta[-1], meta[0], 'bottle_test_refined.png')
                print(meta)
                input()


def check_err():
    bottle_list = torch.load('bottle_list.pt')
    rot_path = 'bottle_err_rot.pt'
    trans_path = "bottle_err_trans.pt"
    rot_err = torch.load(rot_path)
    trans_err = torch.load(trans_path)
    for i in range(855):
        if rot_err['anno&refined_inter'][i] > 20:
            print(i)

            for key in rot_err:
                print(key, rot_err[key][i])
    for key in rot_err:
        draw_histo(rot_err[key], key + '(rot)')
        draw_histo(trans_err[key], key + "(trans)")
        print(key, np.mean(rot_err[key]), np.std(rot_err[key]),
              np.mean(trans_err[key]), np.std(trans_err[key]),
              )
        print(np.std(np.concatenate([rot_err[key], -rot_err[key]])),
              np.std(np.concatenate([trans_err[key], -trans_err[key]])))


if __name__ == "__main__":
    test_path_list = [
        "./2.npy",
        "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C3/N36/S116/s2/T2/align_depth/90.png",
        "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C3/N36/S116/s2/T2/2Dseg/mask/00090.png",
        "/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C3/N36/S116/s2/T2/objpose/90.json",
        (
            '/mnt/8T/HOI4D_CAD_Model/part_annotations/笔记本电脑/' + "041" + "/objs/new-1-align.obj",
            '/mnt/8T/HOI4D_CAD_Model/mobility_annotations/笔记本电脑/' + "041" + '/objs/new-2.obj',
            '/mnt/8T/HOI4D_CAD_Model/mobility_annotations/笔记本电脑/' + "041" + '/objs/new-1.obj',
            '/mnt/8T/HOI4D_CAD_Model/mobility_annotations/笔记本电脑/' + "041" + "/mobility_v2.json"
        ),
        '/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C3/N36/S116/s2/T2/align_image/90.jpg',
        '/mnt/8T/HOI4D_data_yiqi/ZY20210800002/H2/C3/N36/S116/s2/T2/align_image/90.jpg'
    ]
    # load_articulated_object_paths(test_path_list)
    check_err()