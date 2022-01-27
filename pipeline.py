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
    new_list = []
    cnt = 0
    ban_obj = None
    for path_lists in every_path_generator_from_total_json('total.json'):
        cnt += 1
        if cnt == 1:
            ban_obj = path_lists[-3]
            continue
        if ban_obj in path_lists :
            #
            continue
        all_list.append(path_lists)
    torch.save(all_list, 'all_list.pt')


def optim_pipeline(cuda_num=0, start_pos=0):
    print('Start:cuda=', cuda_num)
    with torch.cuda.device(cuda_num):
        from model import PlainObj
        out_path = '/home/yunze/pose3d_lkb/output/'
        all_list = torch.load('all_list.pt')
        for j, path_lists in enumerate(all_list):
            if j < start_pos:
                continue
            new_out_path = path_lists[0][5].replace('20211110', 'seg_3dnew/seg_3d')
            if not os.path.exists(new_out_path):
                new_out_path = path_lists[0][5]
            inter_rot, inter_trans = get_all_poses_from_0json_path_and_output_log_path(path_lists[0][3], new_out_path)
            if inter_rot is None:
                continue
            fin_all_rot, fin_all_trans = [], []
            # try:
            for i, path_list in tqdm(enumerate(path_lists)):
                # print(len(inter_rot), len(path_lists))
                # input()
                # if i != 120:
                #     continue
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
            # except RuntimeError as e:
            #     print('err', e)
            #     with open('/home/yunze/pose3d_lkb/output/err.txt', 'a') as f:
            #         f.write(str(j) + '\n')


def check_single_pipeline(target):
    akoga = torch.load('all_list.pt')
    new_path_list = []
    for path_lists in akoga:
        if path_lists[0][1].find(target) != -1:
            new_path_list = path_lists[0]
            model_input = path_list2plainobj_input(*new_path_list[:5])
            vis3d(model_input[0], model_input[1], new_path_list[-1], new_path_list[1], new_path_list[-3],
                  new_path_list[0], model_input[-1], new_path_list[3])
            break
    else:
        print('hehe')
        return
    from model import PlainObj
    model_input = path_list2plainobj_input(*new_path_list[:5])
    init_model = PlainObj(*model_input)
    init_model.to("cuda:0")
    with torch.no_grad():
        _, _, _, dep, seg = init_model()
        vis2d(dep, seg, new_path_list[-1], model_input[-1], './bottle_check_new3.jpg')
        # vis3d(model_input[0], model_input[1], new_path_list[-1], new_path_list[1], new_path_list[-3], new_path_list[0], model_input[-1], new_path_list[3])


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


if __name__ == "__main__":
    load_articulated_object()