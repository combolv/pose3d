import torch
import torch.nn as nn
import numpy as np
import torchgeometry
import soft_renderer.functional as srf
from torch import optim
from chamfer_distance import ChamferDistance
import os


# class GlobalRotTemporal(nn.Module):
#     pass


class ObjPose(nn.Module):
    """
    Multi-Rigid Object Pose Refine
    R, T, CAD_model, gt_seg, gt_dpt, gt_pcd, ... -> losses, rendered_seg, rendered_dpt
    transformed_v = R v + T
    pcd_loss = CDloss(transformed_v, gt_pcd)
    rendered_seg, rendered_dpt = SoftRas(projected_faces)
    seg_loss = ~HandMask * l1_loss(rendered_seg, gt_seg)
    dpt_loss = ~HandMask * l2_loss(rendered_dpt, gt_dpt)
    """

    def __init__(self, rotvec, trans, vertices, triangles, objseg, handMask, objdpt, pcd_list, camMat, crop_list,
                 gt_frame = (0, 9)):
        super(ObjPose, self).__init__()
        self.frame_len = len(rotvec)
        self.gt_frame = gt_frame
        assert self.frame_len == len(trans)
        assert self.frame_len == len(objseg)
        assert self.frame_len == len(handMask)
        assert self.frame_len == len(objdpt)
        assert self.frame_len == len(pcd_list)
        assert self.frame_len == len(crop_list)
        self.rotvec = nn.Parameter(torch.FloatTensor(rotvec))
        self.trans = nn.Parameter(torch.FloatTensor(trans))
        self.gt_rotvec = nn.Parameter(torch.FloatTensor(rotvec[gt_frame, :]))
        self.gt_trans = nn.Parameter(torch.FloatTensor(trans[gt_frame, :]))

        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))

        # self.vec2mat = torchgeometry.angle_axis_to_rotation_matrix()
        self.pcdloss = ChamferDistance()

        self.seg = nn.Parameter(torch.FloatTensor(objseg)[..., 0])
        self.seg.requires_grad = False
        self.dpt = nn.Parameter(torch.stack([torch.FloatTensor(objdpt)] * 3, dim=-1))
        self.dpt.requires_grad = False
        self.msk = nn.Parameter(torch.FloatTensor(handMask))
        self.msk.requires_grad = False
        self.pcd = []

        for pcds in pcd_list:
            self.pcd.append(nn.Parameter(torch.FloatTensor(pcds)))
            self.pcd[-1].requires_grad = False

        f = int(np.round(camMat[0, 0]))
        self.crop = []
        for crop in crop_list:
            x1, x2, y1, y2 = crop
            self.crop.append([x1 - int(np.round(camMat[1, 2])) + f, x2 - int(np.round(camMat[1, 2])) + f,
                         y1 - int(np.round(camMat[0, 2])) + f, y2 - int(np.round(camMat[0, 2])) + f])

        self.imsize = f * 2


    def forward(self):
        R_inv = torchgeometry.angle_axis_to_rotation_matrix(self.rotvec)[:, :3, :3].transpose(1, 2)
        # print(self.faces.size(), R_inv.size())
        transformed_faces = torch.einsum("ijk, lkm -> ijlm", self.faces, R_inv) + self.trans
        transformed_faces = transformed_faces.permute(2, 0, 1, 3)
        transformed_vertices = torch.einsum("ij, kjl -> ikl", self.vertices, R_inv) + self.trans
        transformed_vertices = transformed_vertices.transpose(0, 1)

        ret1 = transformed_faces[..., 0] / transformed_faces[..., 2]
        # SoftRas uses a left-hand coordinate system
        ret2 = - transformed_faces[..., 1] / transformed_faces[..., 2]
        ret3 = transformed_faces[..., 2]
        projected_faces = torch.stack([ret1, ret2, ret3], dim=-1)
        # projected_faces = projected_faces.unsqueeze(0)
        depth = torch.stack([projected_faces[..., 2]] * 3, dim=-1)

        render_result = srf.soft_rasterize(projected_faces, depth,
                                           self.imsize, texture_type='vertex', near=0.1, eps=1e-4, )

        render_result = render_result.permute(0, 2, 3, 1)

        rendered_depth = []
        rendered_seg = []
        for i, crop in enumerate(self.crop):
            rendered_depth.append(render_result[i, crop[0]:crop[1], crop[2]:crop[3], :3])
            rendered_seg.append(render_result[i, crop[0]:crop[1], crop[2]:crop[3], 3])

        rendered_depth = torch.stack(rendered_depth)
        rendered_seg = torch.stack(rendered_seg)

        pcd_loss_list = []
        for i in range(self.frame_len):
            assert len(transformed_vertices[i]) > 100
            assert len(self.pcd[i]) > 100
            pcd_now = self.pcd[i].cuda().unsqueeze(0)

            _, pcd_loss = self.pcdloss(transformed_vertices[i].unsqueeze(0), pcd_now)
            pcd_loss_list.append(torch.mean(pcd_loss))


        pcd_loss = torch.sum(torch.stack(pcd_loss_list))
        masked_seg = rendered_seg * self.msk[..., 0]
        seg_sum, seg_prod = masked_seg + self.seg, masked_seg * self.seg
        seg_loss = 1 - torch.mean(torch.abs(seg_prod)) / torch.mean(torch.abs(seg_sum - seg_prod))
        seg_mask = (torch.stack([rendered_seg.detach()] * 3, dim=-1) > 0.5).int()
        dpt_loss = torch.mean(torch.abs(rendered_depth - self.dpt * seg_mask) * self.msk)
        # else:
        #     dpt_loss =
        # temporal_loss = ...
        gt_loss = torch.norm(self.rotvec[self.gt_frame, :] - self.gt_rotvec)
        # qrs = torchgeometry.angle_axis_to_quaternion(self.rotvec)

        thetas = torch.norm(self.rotvec, dim=-1)
        ws = self.rotvec.transpose(0, 1) / thetas
        dthetas = torch.abs(torch.sum(ws[:, 1:] * ws[:, :-1], dim=0))
        temporal_rot = torch.sum(torch.square(thetas[2:] + thetas[:-2] - 2 * thetas[1:-1])) + \
            torch.sum(torch.square(dthetas[1:] - dthetas[:-1]))
        # temporal_rot = torch.norm(qrs[2:, :] + qrs[:-2, :] - 2 * qrs[1:-1, :])
        # print(self.trans, self.trans[2:, :] + self.trans[:-2, :] - 2 * self.trans[1:-1, :])
        # input()
        temporal_trans = torch.sum(torch.square(self.trans[2:, :] + self.trans[:-2, :] - 2 * self.trans[1:-1, :]))
        return pcd_loss, seg_loss, dpt_loss, gt_loss, temporal_rot, temporal_trans, rendered_seg, rendered_depth[..., 0]


class PlainObjPathList:
    def __init__(self, rgb_folder, mask_folder, depth_folder, CADv, CADt, json0, cam_out, cam_in, action):
        self.rgb_folder = rgb_folder
        self.mask_folder = mask_folder
        self.depth_folder = depth_folder
        self.CADv = CADv
        self.CADt = CADt
        self.json0 = json0
        self.action = action
        self.cam_in = cam_in
        self.cam_out = cam_out


class ObjCache:
    pass


def get_CADvt_and_camIN0316(root):
    mapping = [
        '/nas/HOI4D_ObjPose_Cache/cad/',
        '玩具车',
        '马克杯',
        '笔记本电脑',
        '储物家具',
        '饮料瓶',
        '保险柜',
        '碗',
        '水桶',
        '剪刀',
        '锤子',
        '钳子',
        '水壶',
        '刀',
        '垃圾桶',
        '快递盒',
        '拉杆箱',
        '台灯',
        '订书机',
        '冰箱',
        '椅子'
    ]
    C_idx = root.index('C')
    try:
        C_idx = int(root[C_idx + 1:C_idx + 3])
    except:
        C_idx = int(root[C_idx + 1])
    N_idx = root.index('N')
    try:
        N_idx = int(root[N_idx + 1:N_idx + 3])
    except:
        N_idx = int(root[N_idx + 1])

    H_idx = root.index("80000") + 5
    H_idx = int(root[H_idx])

    out_file_name = os.path.join(mapping[0], mapping[C_idx], str(N_idx).zfill(3) + ".obj")
    return out_file_name + "v.npy", out_file_name + "t.npy", "/home/yunze/pose3d_lkb/" + str(H_idx) + ".npy"


def root_path2PlainObjPathList0316(root):
    base_path = "/mnt/HOI4D_data_20220315/20220314交付"
    mask_base = "/mnt/HOI4D_data_20220315_2Dmask/2D分割二期"
    v_path, t_path, cam_in = get_CADvt_and_camIN0316(root)
    rgb_folder = os.path.join(base_path, root, 'align_image')
    depth_folder = os.path.join(base_path, root, "align_depth")
    cam_out = os.path.join(base_path, root, "3Dseg", "output.log")
    action = os.path.join(base_path, root, "action", "color.json")
    mask_folder = os.path.join(mask_base, root, "2Dseg", "mask")
    json0_path = os.path.join(base_path, root, "objpose", "0.json")
    return PlainObjPathList(rgb_folder, mask_folder, depth_folder, v_path, t_path, json0_path, cam_out, cam_in, action)
    # torch.save(to_save, out)


def mask_preprocess():
    pass


if __name__ == "__main__":
    from loadfile import path_list2plainobj_input
    from interp import get_all_poses_from_0json_path_and_output_log_path
    # from model import PlainObj
    bottle_list = torch.load("bottle_list.pt")
    for path_lists in bottle_list:
        model_inputs = []
        inter_rot, inter_trans = get_all_poses_from_0json_path_and_output_log_path(path_lists[0][3], path_lists[0][5])
        for i, path_list in enumerate(path_lists):
            model_input = path_list2plainobj_input(*path_list[:5], inter_rot[i], inter_trans[i])
            model_inputs.append(model_input)
            if i == 10:
                break
        """
        self, rotvec, trans, vertices, triangles, objseg, handMask, objdpt, pcd_list, camMat, crop_list
        """
        fl = 10
        overall_model_inputs = [np.array(inter_rot[:fl]), np.array(inter_trans[:fl]), model_inputs[0][2], model_inputs[0][3],
                                np.stack([model_inputs[i][4] for i in range(fl)]),
                                np.stack([model_inputs[i][5] for i in range(fl)]),
                                np.stack([model_inputs[i][6] for i in range(fl)]),
                                [model_inputs[i][7] for i in range(fl)],
                                model_inputs[0][8],
                                [model_inputs[i][9] for i in range(fl)]
                                ]
        model = ObjPose(*overall_model_inputs)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        for _ in range(150):
            optimizer.zero_grad()
            pcd_loss, seg_loss, dpt_loss, gt_loss, tmp_rot, tmp_trans, dep, seg = model()
            all_loss = pcd_loss * 100 + seg_loss + 0.2 * dpt_loss + 20 * tmp_trans + 50 * tmp_rot # / seg_loss.item()  + dpt_loss / dpt_loss.item()
            all_loss.backward()
            optimizer.step()
            print(pcd_loss.item(), seg_loss.item(), dpt_loss.item(),
                  gt_loss.item(), tmp_rot.item(), tmp_trans.item())