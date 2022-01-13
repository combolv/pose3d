import torch
# from tqdm import tqdm
from torch import optim
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rt
from loadfile import annoed_path_generator_from_total_json, every_path_generator_from_total_json

from loadfile import path_list2plainobj_input
from interp import get_all_poses_from_0json_path_and_output_log_path
from interp import get_large_gap_poses_from_0json_path_and_output_log_path
from model import PlainObj
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
    for path_lists in every_path_generator_from_total_json('total.json'):
        new_list.append(path_lists)
        cnt += 1
        if cnt % 31 == 0:
            all_list.append(list(new_list.copy()))
    torch.save(all_list, 'all_list.pt')
# def check_specific_path_list(path_list):
#     from vis import vis
#     from loadfile import read_rt
#     r, t = read_rt(path_list[3])
#     vis(r, t, path_list[-1], path_list[1], path_list[-3], path_list[0])


if __name__ == "__main__":
    check_gt_loss_terms_gap10()