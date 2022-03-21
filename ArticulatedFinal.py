import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import torchgeometry
from scipy.spatial.transform import Rotation as Rt

import numpy as np
import json
import argparse
import cv2
import open3d as o3d
import copy

import soft_renderer.functional as srf
# from emd import EMDLoss
from chamfer_distance import ChamferDistance
import os


class SingleAxisObject:
    pass


class AxisMeta:
    def __init__(self):
        pass


class invRodrigues(nn.Module):
    """
    Rodrigues Algorithm
    Axis-Angle -> SO(3) matrix.T
    R = I + sin(t) * [w] + (1 - cos(t)) * [w]^2
    """
    def __init__(self):
        super(invRodrigues, self).__init__()
        # set self.I to be model para is necessary,
        # by model.to(), self.I and rotvec should be automatically in the same device
        self.I = nn.Parameter(torch.eye(3))
        self.I.requires_grad = False

    def forward(self, rotvec):
        # Rodrigues algorithm: validated by scipy.spatial
        theta = torch.norm(rotvec)
        omegaMat = torch.stack([torch.cross(self.I[i], rotvec) for i in range(3)]) / theta
        R = self.I + torch.sin(theta) * omegaMat + \
            (1 - torch.cos(theta)) * torch.mm(omegaMat, omegaMat)
        return R.transpose(0, 1)


class projCamera(nn.Module):
    """
    Perspective Camera Projection
    camera coordinate -> pixel coordinate
    x = x / z
    y = - y / z
    """
    def __init__(self):
        super(projCamera, self).__init__()

    def forward(self, mesh):
        ret1 = mesh[..., 0] / mesh[..., 2]
        # SoftRas uses a left-hand coordinate system
        ret2 = - mesh[..., 1] / mesh[..., 2]
        ret3 = mesh[..., 2]
        return torch.stack([ret1, ret2, ret3], dim=-1)


class AxisTheta(nn.Module):
    """
    Theta + Axis Annotation -> Inv Rot Mat, Virtual Trans
    """
    def __init__(self, direction_rot_vec, orig, rad_max, rad_min):
        super(AxisTheta, self).__init__()
        self.I = nn.Parameter(torch.eye(3))
        self.I.requires_grad = False
        self.omegaMat = nn.Parameter(direction_rot_vec)
        self.omegaMat.requires_grad = False
        self.orig = nn.Parameter(orig)
        self.orig.requires_grad = False
        self.theta_min, self.theta_max = rad_min, rad_max

    def forward(self, theta):
        R = self.I + torch.sin(theta) * self.omegaMat + \
            (1 - torch.cos(theta)) * torch.mm(self.omegaMat, self.omegaMat)
        virt_trans = self.orig - R @ self.orig
        return R.transpose(0, 1), virt_trans


class FaceVertices(nn.Module):
    """
    wavefront -> transformed CAD model
    __init__: vertices [n, 3], triangles [n, 3] -> vertices [n, 3], faces [n, 3, 3]
    forward: R [3, 3], t [3] -> transformed vertices [1, n, 3], faces [1, n, 3, 3], depth [1, n, 3, 3]
    """

    def __init__(self, vertices, triangles):
        super(FaceVertices, self).__init__()
        self.cam2pix = projCamera()
        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))
        self.faces.requires_grad = False
        self.vertices.requires_grad = False

    def forward(self, R_inv, trans):
        transformed_faces = torch.matmul(self.faces, R_inv) + trans
        transformed_vertices = torch.matmul(self.vertices, R_inv) + trans
        projected_faces = self.cam2pix(transformed_faces)
        projected_faces = projected_faces.unsqueeze(0)
        depth = torch.stack([projected_faces[..., 2]] * 3, dim=-1)
        return transformed_vertices.unsqueeze(0), projected_faces, depth


class SegDptLossBatch(nn.Module):
    def __init__(self):
        super(SegDptLossBatch, self).__init__()

        # TODO: move seg, dpt, msk init here
        # TODO: register
        # TODO: bug fixed(check by o3d)

    def forward(self, batch_seg, batch_depth, batch_msk, gt_seg):
        masked_seg = batch_seg * batch_msk[..., 0]
        seg_sum, seg_prod = masked_seg + gt_seg, masked_seg * gt_seg
        seg_loss = 1 - torch.mean(torch.abs(seg_prod), (1, 2)) / torch.mean(torch.abs(seg_sum - seg_prod), (1, 2))
        seg_mask = (torch.stack([batch_seg.detach()] * 3, dim=-1) > 0.5).int()
        # dpt_loss = torch.mean(torch.abs(batch_depth - batch_depth * seg_mask) * batch_msk, keepdim=0)
        return seg_loss, None


class SoftRasLayer(nn.Module):
    """

    """

    def __init__(self, image_size, crop_list):
        super(SoftRasLayer, self).__init__()
        self.imsize = image_size
        self.crop = crop_list

    def forward(self, faces, depth):
        render_result = srf.soft_rasterize(faces, depth, self.imsize, texture_type='vertex',
                                           near=0.1, eps=1e-4, gamma_val=1e-5, dist_eps=1e-5)
        render_result = render_result.squeeze(0).permute((1, 2, 0))
        rendered_depth = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], :3]
        rendered_seg = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], 3]

        return rendered_seg, rendered_depth


class ArtObj(nn.Module):
    """
    Support Articulated Object
    {
        axis (class containing relative position, direction, limit),
        parameters (7DoF &/| 2 * 6DoF),
        meshes (vertices + triangles (simplified) : base, part, aligned part),
        cropped masks (base, part, overall, mask of region occluded by hand or no depth, hand + part, hand + base),
        cropped depth image,
        3d-cropped depth point cloud (base, part, both),
        camera intrinsics,
        crop position in (1920, 1080),
        method = 0: separately, 1: globally, 2: both, 3: base unsupervised, 4: part unsupervised, 5: theta fixed
    } -->
    {
        depth_loss : List[both pcd loss, base pcd loss, part pcd loss], (may contain None)
        seg_loss : Tensor [1, 3], pcd_loss, rendered_depth, rendered_seg
    }

    """
    SEP12DoF, GLB7DoF, BOTH, THT_FIX = list(range(4))

    def __init__(self, axis_meta, para_meta, mesh_meta, mask_meta, depth2d, pcd, camMat, crop_list, method=2):
        super(ArtObj, self).__init__()
        self.method = method
        # direction_rot_vec, orig, rad_max, rad_min = axis_meta
        # process parameters
        self.para = nn.Parameter(torch.FloatTensor(para_meta[:-1]))
        self.theta = nn.Parameter(torch.FloatTensor([para_meta[-1]]))
        if method == ArtObj.THT_FIX:
            self.theta.requires_grad = False
        self.vec2mat = invRodrigues()

        # load point cloud
        if len(pcd[0]) < 10:
            self.pcd0 = None
        else:
            self.pcd0 = nn.Parameter(torch.FloatTensor(pcd[0]))
            self.pcd0.requires_grad = False
        if len(pcd[1]) < 10:
            self.pcd1 = None
        else:
            self.pcd1 = nn.Parameter(torch.FloatTensor(pcd[1]))
            self.pcd1.requires_grad = False
        self.pcd = nn.Parameter(torch.FloatTensor(pcd[2]))
        self.pcd.requires_grad = False

        # load mesh
        self.mesh_base = FaceVertices(mesh_meta[0], mesh_meta[1])
        self.mesh_part = FaceVertices(mesh_meta[2], mesh_meta[3])
        self.mesh_part_canonical = FaceVertices(mesh_meta[4], mesh_meta[5])

        # cropped 2d images
        self.seg = nn.Parameter(torch.FloatTensor(mask_meta[:3])[..., 0])
        self.dpt = nn.Parameter(torch.stack([torch.FloatTensor(depth2d)] * 3, dim=-1))
        self.msk = nn.Parameter(torch.FloatTensor(mask_meta[3:]))
        self.seg.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False

        # external modules
        self.axis = AxisTheta(*axis_meta)
        self.vec2mat = invRodrigues()
        self.pcdloss = ChamferDistance()
        self.criterion2d = SegDptLossBatch()

        # size info
        f = int(np.round(camMat[0, 0]))
        x1, x2, y1, y2 = crop_list
        self.crop = [x1 - int(np.round(camMat[1, 2])) + f, x2 - int(np.round(camMat[1, 2])) + f,
                     y1 - int(np.round(camMat[0, 2])) + f, y2 - int(np.round(camMat[0, 2])) + f]

        self.srf_layer = SoftRasLayer(2 * f, self.crop)

    def forward(self):
        # print(self.para[0].size())
        base_Rinv = self.vec2mat(self.para[0])
        base_verts, base_faces, base_depth = self.mesh_base(base_Rinv, self.para[1])
        part_Rinv_canonical = self.vec2mat(self.para[2])
        part_verts_canonical, part_faces_canonical, part_depth_canonical = self.mesh_part_canonical(part_Rinv_canonical,
                                                                                                    self.para[3])
        # print(self.theta.size())
        part_rel_Rinv, virt_trans = self.axis(self.theta)
        part_Rinv = torch.mm(part_rel_Rinv, base_Rinv)
        part_trans = base_Rinv.transpose(0, 1) @ virt_trans + self.para[1]
        part_verts, part_faces, part_depth = self.mesh_part(part_Rinv, part_trans)

        both_faces = torch.cat([base_faces, part_faces], dim=1)
        both_depth = torch.cat([base_depth, part_depth], dim=1)
        # self.srf_layer.crop = [0, 3000, 0, 3000]
        base_seg, base_depth2d = self.srf_layer(base_faces, base_depth)
        both_seg, both_depth2d = self.srf_layer(both_faces, both_depth)
        part_seg_canonical, part_depth2d_canonical = self.srf_layer(part_faces_canonical, part_depth_canonical)
        #
        all_seg = torch.stack([base_seg, part_seg_canonical, both_seg])
        # print(torch.min(self.mesh_base.vertices, dim=0), torch.max(self.mesh_base.vertices, dim=0))
        # print(self.para[0], self.para[1])
        # print(base_seg.size(), base_depth2d.size())
        # print(torch.min(base_faces[..., 0]), torch.max(base_faces[..., 0]))
        # print(torch.min(base_faces[..., 1]), torch.max(base_faces[..., 1]))
        # print(torch.min(base_faces[..., 2]), torch.max(base_faces[..., 2]))
        # print(self.srf_layer.crop)
        all_depth2d = torch.stack([base_depth2d, part_depth2d_canonical, both_depth2d])

        seg_loss, depth_loss = self.criterion2d(all_seg, all_depth2d, self.msk, self.seg)

        # if self.method == self.SEP12DoF:

        _, pcd_loss = self.pcdloss(torch.cat((base_verts, part_verts), dim=1), self.pcd.unsqueeze(0))
        # else:
        #     pcd_loss = None
        # if self.pcd0 is not None and self.method != self.GLB7DoF:
        pcd_loss = torch.mean(pcd_loss)
        if self.pcd0 is not None:
            _, pcd_base = self.pcdloss(base_verts, self.pcd0.unsqueeze(0))
            pcd_base = torch.mean(pcd_base)
            pcd_loss = pcd_loss + pcd_base
        else:
            pcd_base = None
        # if self.pcd1 is not None and self.method != self.GLB7DoF:
        pcd_cons = None
        if self.pcd1 is not None:
            _, pcd_part = self.pcdloss(part_verts_canonical, self.pcd1.unsqueeze(0))
            pcd_part = torch.mean(pcd_part)
            pcd_cons1, pcd_cons2 = self.pcdloss(part_verts_canonical, part_verts)
            pcd_loss = pcd_loss + pcd_part
            pcd_cons = torch.mean(pcd_cons1) + torch.mean(pcd_cons2)
        else:
            pcd_part = None


        # if self.method == self.BOTH:
        #     _, constraint_loss = self.pcdloss(self.pcd.unsqueeze(0), self.pcd1.unsqueeze(0))
        """
        if self.method == self.BOTH:
            # part_Rinv approx part_Rinv_canonical
            torch.trace(torch.mm(part_Rinv, part_Rinv_canonical.transpose(0, 1)))
            constraint_loss_rot = torch.arccos((torch.trace(torch.mm(part_Rinv, part_Rinv_canonical.transpose(0, 1))) - 1) / 2)
            constraint_loss_trans = torch.square()
        else:
            constraint_loss = None

        # Need Umeyama algo. to find rel. pose from can. to part
        """
        # seg_loss = torch.sum(seg_loss)
        # print(pcd_loss.size(), seg_loss.size())
        return [pcd_loss, pcd_base, pcd_part, pcd_cons], \
               seg_loss, depth_loss, all_seg, all_depth2d[..., 0]


def articulated_pt_file2model_input(pt_file):
    pt_obj = torch.load(pt_file)


def articulated_obj_all_pose(json0_path, out_path):
    from interp import get_all_poses_from_0json_path_and_output_log_path
    all_rots, all_transis = get_all_poses_from_0json_path_and_output_log_path(json0_path, out_path, num=0)
    part_rots, part_trans = get_all_poses_from_0json_path_and_output_log_path(json0_path, out_path, num=1)
    base_rot_matrices = Rt.from_rotvec(all_rots).as_matrix()
    part_rot_matrices = Rt.from_rotvec(part_rots).as_matrix()
    part_z = part_rot_matrices[:, :, 2]
    base_y, base_z = base_rot_matrices[:, :, 1], base_rot_matrices[:, :, 2]
    signs = np.sign(np.sum(part_z * base_y, axis=-1))
    thetas = np.arccos(np.sum(part_z * base_z, axis=-1))
    # print(thetas * 180 / np.pi, signs)
    # input()
    return all_rots, all_transis, part_rots, part_trans, thetas * signs


def vis3darticulated(r, T, part_r, part_T, rgb_path, depth_path, cam_in_path,
                     base_v, base_idx, part_v, part_idx,
                     crop_list = None, json0_path = None, outsrc_pcd = None):
    def create_geo_from_vt(vertices, triangles):
        # vertices, triangles = vertices.numpy(), triangles.numpy()
        # print(np.max(vertices.reshape((-1, 3)), axis=0) - np.min(vertices.reshape((-1, 3)), axis=0))
        return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_crop = np.zeros_like(depth_raw)
    if crop_list is not None:
        depth_crop[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]] = depth_raw[crop_list[0]:crop_list[1], crop_list[2]:crop_list[3]]
    else:
        depth_crop = depth_raw
    depth_raw = o3d.geometry.Image(depth_crop)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_raw, depth_raw, convert_rgb_to_intensity=False )

    camMat = np.load(cam_in_path)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2]))

    cad = create_geo_from_vt(base_v, base_idx)
    cad2 = create_geo_from_vt(part_v, part_idx)

    R = [Rt.from_rotvec(r).as_matrix(), Rt.from_rotvec(part_r).as_matrix()]
    trans_list = [T, part_T]
    mesh_mv = copy.deepcopy(cad).translate(T)
    mesh_mv.rotate(R[0], center=T)
    mesh_mv2 = copy.deepcopy(cad2).translate(part_T)
    mesh_mv2.rotate(R[1], center=part_T)
    prt_list = [pcd, mesh_mv, mesh_mv2]
    # print(len(pcd.points))
    if json0_path is not None:
        from loadfile import read_rtd
        for num in range(2):
            _, _, D = read_rtd(json0_path, num=num)
            box = o3d.geometry.OrientedBoundingBox(center=trans_list[num], R=R[num], extent=D)
            pc = box.get_box_points()
            rft = o3d.geometry.LineSet.create_from_oriented_bounding_box(o3d.geometry.OrientedBoundingBox(
                center=(np.asarray(pc[6]) + np.asarray(pc[4])) / 2,
                R=R[num], extent=np.array([0.01, 0.01, 0.01])))
            rft.paint_uniform_color(np.array([1., 0., 0.]))
            prt_list.extend([box, rft])
    if outsrc_pcd is not None:
        prt_list.append(outsrc_pcd)
        prt_list = prt_list[-3:]

    o3d.visualization.draw_geometries(prt_list)


def path_list2artobj_input(cam_in_path, dpt_path, mask_path, pt_file_path):
    from loadfile import read_mask2bbox
    from loadfile import read_mask
    pt_obj = torch.load(pt_file_path)
    depth2d = cv2.imread(dpt_path, cv2.IMREAD_UNCHANGED)
    camMat = np.load(cam_in_path)
    x1, x2, y1, y2, clean_mask = read_mask2bbox(mask_path, obj_color=3)
    crop_list = [x1, x2, y1, y2]

    depth2d = np.array(depth2d, dtype=np.float32) / 1000
    obj_base_mask, hand_mask = read_mask(clean_mask, dscale=1, crop=crop_list)
    large_mask_base, _ = read_mask(clean_mask, dscale=1)
    large_mask_part, _ = read_mask(clean_mask, obj_color=3, dscale=1)
    large_mask_overall = large_mask_part + large_mask_base
    obj_part_mask = large_mask_part[x1:x2, y1:y2, :]
    obj_overall_mask = obj_base_mask + obj_part_mask
    depth3d_base = o3d.geometry.Image(depth2d * large_mask_base[..., 0])
    depth3d_part = o3d.geometry.Image(depth2d * large_mask_part[..., 0])
    depth3d_all = o3d.geometry.Image(depth2d * large_mask_overall[..., 0])
    depth2d = depth2d[x1:x2, y1:y2]
    hand_mask[np.where(depth2d < 0.001)] = 0
    hand_and_part = hand_mask * ( 1 - obj_part_mask )
    hand_and_base = hand_mask * ( 1 - obj_base_mask )

    # load point cloud from depth
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])
    pcd_part = o3d.geometry.PointCloud.create_from_depth_image(depth3d_part, intrinsics, stride=2)
    pcd_base = o3d.geometry.PointCloud.create_from_depth_image(depth3d_base, intrinsics, stride=2)
    pcd_all = o3d.geometry.PointCloud.create_from_depth_image(depth3d_all, intrinsics, stride=2)
    pcd_part = np.asarray(pcd_part.points)
    pcd_base = np.asarray(pcd_base.points)
    pcd_all = np.asarray(pcd_all.points)

    # axis_meta = Axis(mob_path)
    # axis_meta.orig += base_translation
    # para_meta = [rot_base, trans_base, rot_part, trans_part, theta]
    mesh_meta = [pt_obj.base_verts, pt_obj.base_idx, pt_obj.part_verts, pt_obj.part_idx, pt_obj.part_align_verts,
                 pt_obj.part_align_idx]
    axis_meta = [torch.FloatTensor(Rt.from_rotvec(pt_obj.axis_rot).as_matrix()), pt_obj.virt_p,
                 pt_obj.theta_lim_maximum, pt_obj.theta_lim_minimum]
    """
    self.base_verts = torch.FloatTensor(base_verts)
        self.base_idx = torch.LongTensor(base_idx)
        self.part_verts = torch.FloatTensor(part_verts)
        self.part_idx = torch.LongTensor(part_idx)
        self.part_align_verts = torch.FloatTensor(part_align_verts)
        self.part_align_idx = torch.LongTensor(part_align_idx)
        self.rel_se3 = torch.FloatTensor(se3)
        self.axis_rot = torch.FloatTensor(axis_dir)
        self.virt_p = torch.FloatTensor(axis_orig) # virtual trans = p - R @ p, R = axisAngle2mat(axis_rot * theta)
        self.theta_lim_maximum = torch.FloatTensor([degree_max / 180 * np.pi])
        self.theta_lim_minimum = torch.FloatTensor([degree_min / 180 * np.pi])
    """
    mask_meta = [obj_base_mask, obj_part_mask, obj_overall_mask, hand_and_part, hand_and_base, hand_mask]
    pcd_list = [pcd_base, pcd_part, pcd_all]

    return [axis_meta, None, mesh_meta, mask_meta, depth2d, pcd_list, camMat, crop_list]


if __name__ == "__main__":
    from vis import vis2d
    mask_base = base_path = "/nas/datasets/HOI4D_data_yiqi_20220311/一期年前交付/ZY20210800004/H4/C8/N3/S71/s1/T2/"
    # base_path = "/mnt/HOI4D_data_20220315/20220314交付/ZY20210800001/H1/C8/N09/S315/s05/T1/"
    # mask_base = "/mnt/HOI4D_data_20220315_2Dmask/2D分割二期/ZY20210800001/H1/C8/N09/S315/s05/T1/"
    all_rot, all_trans, part_rot, part_trans, thetas = articulated_obj_all_pose(
        "./bucket3/0.json",
        # "/mnt/HOI4D_data_20220315/20220314交付/ZY20210800001/H1/C8/N09/S315/s05/T1/objpose/0.json",
        os.path.join(base_path, "3Dseg", "output.log")
    )
    for frame_id in range(300):
        art_input = path_list2artobj_input('./4.npy', os.path.join(base_path, "align_depth", str(frame_id) + ".png"),
                               os.path.join(mask_base, "2Dseg", "mask", str(frame_id).zfill(5) + ".png"),
                               "/nas/HOI4D_ObjPose_Cache/bucket_test/003.pt")
        pt_obj = torch.load("/nas/HOI4D_ObjPose_Cache/bucket_test/003.pt")

        # vis3d(all_rot[frame_id], all_trans[frame_id], os.path.join(base_path, "align_image", str(frame_id) + '.jpg'),
        #       os.path.join(base_path, "align_depth", str(frame_id) + ".png"), "test.obj", './4.npy',
        #       json0_path="./bucket3/0.json", crop_list=art_input[-1])

        # vis3darticulated(all_rot[frame_id], all_trans[frame_id], part_rot[frame_id], part_trans[frame_id],
        #                  os.path.join(base_path, "align_image", str(frame_id) + '.jpg'),
        #                  os.path.join(base_path, "align_depth", str(frame_id) + ".png"), './4.npy',
        #                 pt_obj.base_verts, pt_obj.base_idx, pt_obj.part_align_verts, pt_obj.part_align_idx, crop_list=art_input[-1],
        #                  json0_path="./bucket3/0.json" # TODO: add theta and draw double box!
        #                  )

        i = frame_id
        art_input[1] = [all_rot[i], all_trans[i], part_rot[i], part_trans[i], thetas[i]]
        art_model = ArtObj(*art_input)
        art_model.cuda()
        pcd_loss, seg_loss, dpt_loss, dep, seg = art_model()

        for j in range(3):
            try:
                vis2d(dep[j], seg[j], os.path.join(base_path, "align_image", str(frame_id) + '.jpg'), art_input[-1],
                      'bucket_test0320/bucket_before{}_{}.png'.format(frame_id, j))
            except:
                continue
        optimizer = optim.Adam(art_model.parameters(), lr=0.001)
        all_losses = {
            "pcd0": [],
            "cons": [],
            "seg0": [],
            "seg1": [],
            "seg2": [],
        }
        for _ in range(150):
            optimizer.zero_grad()
            pcd_loss, seg_loss, dpt_loss, dep, seg = art_model()
            all_loss = pcd_loss[0] * 50 + seg_loss[0] + seg_loss[1] + seg_loss[2] #+ 0.2 * dpt_loss   / seg_loss.item()  + dpt_loss / dpt_loss.item()
            if pcd_loss[-1] is not None:
                all_loss = all_loss + pcd_loss[-1] * 50
                print("cons:", pcd_loss[-1].item())
                all_losses["cons"].append(pcd_loss[-1].item())
            else:
                all_losses["cons"].append(None)
            all_loss.backward()
            optimizer.step()
            all_losses["pcd0"].append(pcd_loss[0].item())
            for j in range(3):
                all_losses["seg"+str(j)].append(seg_loss[j].item())
            print(pcd_loss[0].item(), seg_loss[0].item(), seg_loss[1].item(), seg_loss[2].item())
        for j in range(3):
            try:
                vis2d(dep[j], seg[j],  os.path.join(base_path, "align_image", str(frame_id) + '.jpg'), art_input[-1], 'bucket_test0320/bucket_after{}_{}.png'.format(frame_id,j))
            except:
                continue
        torch.save(all_losses, "./bucket_test0320/loss{}.pt".format(frame_id))
        np.save("./bucket_test0320/theta{}.npy".format(frame_id), art_model.theta.detach().cpu().numpy())
        np.save("./bucket_test0320/para{}.npy".format(frame_id), art_model.para.detach().cpu().numpy())
        # vis3d(art_model.para[0].detach().cpu().numpy(), art_model.para[1].detach().cpu().numpy(), os.path.join(base_path, "align_image", str(frame_id) + '.jpg'),
        #       os.path.join(base_path, "align_depth", str(frame_id) + ".png"), "test.obj", './4.npy',
        #       json0_path="./bucket3/0.json")

    # "/nas/datasets/HOI4D_data_yiqi_20220311/一期年前交付/ZY20210800004/H4/C8/N3/S71/s1/T2/2Dseg/overlay/"
    #