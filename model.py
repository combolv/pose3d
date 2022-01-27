from operator import gt
from traceback import print_tb
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import json
import argparse
import cv2
import open3d as o3d

from manopth.manopth.manolayer import ManoLayer
from manopth.manopth import demo

from vis import *

import soft_renderer.functional as srf
# from emd import EMDLoss
from chamfer_distance import ChamferDistance


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


class projKps(nn.Module):
    """
    Perspective Camera Projection
    camera coordinate -> pixel coordinate
    x = fx * x / z
    y = - fy * y / z
    """
    def __init__(self, camMat):
        super(projKps, self).__init__()
        self.xscale = camMat[0, 0] / 1000
        self.yscale = camMat[1, 1] / 1000

    def forward(self, mesh):
        ret1 = self.xscale * mesh[..., 0] / mesh[..., 2]
        # SoftRas uses a left-hand coordinate system
        ret2 = self.yscale * mesh[..., 1] / mesh[..., 2]
        ret3 = mesh[..., 2]
        return torch.stack([ret1, ret2, ret3], dim=-1)


class AxisTheta(nn.Module):
    """
    Theta + Axis Annotation -> Inv Rot Mat, Virtual Trans
    """
    def __init__(self, axis):
        super(AxisTheta, self).__init__()
        self.I = nn.Parameter(torch.eye(3))
        self.I.requires_grad = False
        self.omegaMat = nn.Parameter(torch.FloatTensor(axis.rot_mat))
        self.omegaMat.requires_grad = False
        self.orig = nn.Parameter(torch.FloatTensor(axis.orig))
        self.orig.requires_grad = False
        self.theta_min, self.theta_max = axis.rad_min, axis.rad_max

    def forward(self, theta):
        R = self.I + torch.sin(theta) * self.omegaMat + \
            (1 - torch.cos(theta)) * torch.mm(self.omegaMat, self.omegaMat)
        virt_trans = self.orig - torch.mm(R, self.orig)
        return R.transpose(0, 1), virt_trans


class PlainObj(nn.Module):
    """
    Single Rigid Object Pose Refine
    R, T, CAD_model, gt_seg, gt_dpt, gt_pcd, ... -> losses, rendered_seg, rendered_dpt
    transformed_v = R v + T
    pcd_loss = CDloss(transformed_v, gt_pcd)
    rendered_seg, rendered_dpt = SoftRas(projected_faces)
    seg_loss = ~HandMask * l1_loss(rendered_seg, gt_seg)
    dpt_loss = ~HandMask * l2_loss(rendered_dpt, gt_dpt)
    """
    def __init__(self, rotvec, trans, vertices, triangles, objseg, handMask, objdpt, pcd, camMat, crop):
        super(PlainObj, self).__init__()
        self.rotvec = nn.Parameter(torch.FloatTensor(rotvec))
        self.trans = nn.Parameter(torch.FloatTensor(trans))

        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))
        self.faces.requires_grad = False
        self.vertices.requires_grad = False

        self.vec2mat = invRodrigues()
        self.cam2pix = projCamera()
        self.pcdloss = ChamferDistance()

        self.seg = nn.Parameter(torch.FloatTensor(objseg)[..., 0])
        self.dpt = nn.Parameter(torch.stack([torch.FloatTensor(objdpt)]*3, dim=-1))
        self.msk = nn.Parameter(torch.FloatTensor(handMask))
        self.pcd = nn.Parameter(torch.FloatTensor(pcd))
        self.seg.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False
        self.pcd.requires_grad = False

        f = int(np.round(camMat[0, 0]))
        x1, x2, y1, y2 = crop
        self.crop = [x1 - int(np.round(camMat[1, 2])) + f, x2 - int(np.round(camMat[1, 2])) + f,
                     y1 - int(np.round(camMat[0, 2])) + f, y2 - int(np.round(camMat[0, 2])) + f]

        self.imsize = f * 2

    def forward(self):
        R_inv = self.vec2mat(self.rotvec)
        transformed_faces = torch.matmul(self.faces, R_inv) + self.trans
        transformed_vertices = torch.matmul(self.vertices, R_inv) + self.trans
        projected_faces = self.cam2pix(transformed_faces)
        projected_faces = projected_faces.unsqueeze(0)
        depth = torch.stack([projected_faces[..., 2]]*3, dim=-1)

        render_result = srf.soft_rasterize(projected_faces, depth,
                                        self.imsize, texture_type='vertex', near=0.1, eps=1e-4, )
        render_result = render_result.squeeze(0).permute((1, 2, 0))
        rendered_depth = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], :3]
        rendered_seg = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], 3]

        # print(transformed_vertices.is_cuda, self.pcd.is_cuda)
        # input()
        _, pcd_loss = self.pcdloss(transformed_vertices.unsqueeze(0), self.pcd.unsqueeze(0))
        pcd_loss = torch.mean(pcd_loss)
        masked_seg = rendered_seg * self.msk[:, :, 0]
        seg_sum, seg_prod = masked_seg + self.seg, masked_seg * self.seg
        seg_loss = 1 - torch.mean(torch.abs(seg_prod)) / torch.mean(torch.abs(seg_sum - seg_prod))
        seg_mask = (torch.stack([rendered_seg.detach()] * 3, dim=-1) > 0.5).int()
        dpt_loss = torch.mean(torch.abs(rendered_depth - self.dpt * seg_mask) * self.msk)

        return pcd_loss, seg_loss, dpt_loss, rendered_seg, rendered_depth[:, :, 0]


class BatchObj(nn.Module):
    """
    Support multi-frame inputs
    Use project_faces = torch.stack([project_faces])
    Not implemented yet
    """

    def __init__(self, rotvec_seq, trans_seq, vertices, triangles, objseg_seq, handMask_seq, objdpt_seq, pcd_seq, camMat, crop_seq):
        super(BatchObj, self).__init__()
        self.frame = len(rotvec_seq)
        self.rotvec = nn.Parameter(torch.FloatTensor(rotvec_seq))
        self.trans = nn.Parameter(torch.FloatTensor(trans_seq))

        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))
        self.faces.requires_grad = False
        self.vertices.requires_grad = False

        self.vec2mat = invRodrigues()
        self.cam2pix = projCamera(camMat)
        self.pcdloss = ChamferDistance()

        self.gt_seg, self.gt_dpt, self.gt_msk, self.gt_pcd = [], [], [], []
        for i in range(self.frame):
            seg = nn.Parameter(torch.FloatTensor(objseg_seq[i])[..., 0])
            seg.requires_grad = False
            self.gt_seg.append(seg)
            dpt = nn.Parameter(torch.stack([torch.FloatTensor(objdpt_seq[i])] * 3, dim=-1))
            dpt.requires_grad = False
            self.gt_dpt.append(dpt)
            msk = nn.Parameter(torch.FloatTensor(handMask_seq[i]))
            msk.requires_grad = False
            self.gt_msk.append(msk)
        # self.msk = nn.Parameter(torch.FloatTensor(handMask))
        # self.pcd = nn.Parameter(torch.FloatTensor(pcd))
        self.seg.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False
        self.pcd.requires_grad = False

        # self.crop = crop

    def forward(self):
        R_inv = self.vec2mat(self.rotvec)
        transformed_faces = torch.matmul(self.faces, R_inv) + self.trans
        transformed_vertices = torch.matmul(self.vertices, R_inv) + self.trans
        projected_faces = self.cam2pix(transformed_faces)
        projected_faces = projected_faces.unsqueeze(0)
        depth = torch.stack([projected_faces[..., 2]] * 3, dim=-1)

        render_result = srf.soft_rasterize(projected_faces, depth,
                                           self.imsize, texture_type='vertex', near=0.5)
        render_result = render_result.permute((1, 2, 0))
        rendered_depth = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], :3]
        rendered_seg = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], 3]

        pcd_loss = self.pcdloss(transformed_vertices, self.pcd)
        # pcd_loss = 0.0

        seg_loss = torch.mean(torch.abs(rendered_seg - self.seg) * self.msk[:, :, :, 0])
        # print(rendered_depth.size(), self.dpt.size(), self.msk.size())
        # input()
        dpt_loss = torch.mean(torch.square(rendered_depth - self.dpt) * self.msk)

        return pcd_loss, seg_loss, dpt_loss, rendered_seg, rendered_depth


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

    def forward(self, batch_seg, batch_depth, batch_msk):
        masked_seg = batch_seg * batch_msk[..., 0]
        seg_sum, seg_prod = masked_seg + self.seg, masked_seg * self.seg
        seg_loss = 1 - torch.mean(torch.abs(seg_prod), keepdim=0) / torch.mean(torch.abs(seg_sum - seg_prod), keepdim=0)
        seg_mask = (torch.stack([batch_seg.detach()] * 3, dim=-1) > 0.5).int()
        dpt_loss = torch.mean(torch.abs(batch_depth - self.dpt * seg_mask) * batch_msk, keepdim=0)
        return seg_loss, dpt_loss


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

        # process parameters
        self.para = nn.Parameter(torch.FloatTensor(para_meta[:-1]))
        self.theta = nn.Parameter(torch.FloatTensor([para_meta[-1]]))
        if method == ArtObj.THT_FIX:
            self.theta.requires_grad = False
        self.vec2mat = invRodrigues()

        # load point cloud
        if len(pcd[0] < 10):
            self.pcd0 = None
        else:
            self.pcd0 = nn.Parameter(torch.FloatTensor(pcd[0]))
            self.pcd0.requires_grad = False
        if len(pcd[1] < 10):
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
        self.axis = AxisTheta(axis_meta)
        self.vec2mat = invRodrigues()
        self.pcdloss = ChamferDistance()
        self.criterion2d = SegDptLossBatch()

        # size info
        f = int(np.round(camMat[0, 0]))
        x1, x2, y1, y2 = crop_list
        self.crop = [x1 - int(np.round(camMat[1, 2])) + f, x2 - int(np.round(camMat[1, 2])) + f,
                     y1 - int(np.round(camMat[0, 2])) + f, y2 - int(np.round(camMat[0, 2])) + f]

        self.srf_layer = SoftRasLayer(2 * f, crop_list)


    def forward(self):
        base_Rinv = self.vec2mat(self.para[0])
        base_verts, base_faces, base_depth = self.mesh_base(base_Rinv, self.para[1])
        part_Rinv_canonical = self.vec2mat(self.para[2])
        part_verts_canonical, part_faces_canonical, part_depth_canonical = self.mesh_part_canonical(part_Rinv_canonical, self.para[3])
        part_rel_Rinv, virt_trans = self.axis(self.theta)
        part_Rinv = torch.mm(part_rel_Rinv, base_Rinv)
        part_trans = torch.mm(base_Rinv.transpose(0, 1), virt_trans) + self.para[1]
        part_verts, part_faces, part_depth = self.mesh_part(part_Rinv, part_trans)

        both_faces = torch.cat([base_faces, part_faces])
        both_depth = torch.cat([base_depth, part_depth])
        base_seg, base_depth2d = self.srf_layer(base_faces, base_depth)
        both_seg, both_depth2d = self.srf_layer(both_faces , both_depth)
        part_seg_canonical, part_depth2d_canonical = self.srf_layer(part_faces_canonical, part_depth_canonical)

        all_seg = torch.cat([base_seg, part_seg_canonical, both_seg])
        all_depth2d = torch.cat([base_depth2d, part_depth2d_canonical, both_depth2d])
        seg_loss, depth_loss = self.criterion2d(all_seg, all_depth2d)

        if self.method == self.SEP12DoF:
            _, pcd_loss = self.pcdloss(torch.cat((base_verts, part_verts), dim=1), self.pcd.unsqueeze(0))
        else:
            pcd_loss = None
        if self.pcd0 is not None and self.method != self.GLB7DoF:
            _, pcd_base = self.pcdloss(base_verts, self.pcd0.unsqueeze(0))
        else:
            pcd_base = None
        if self.pcd1 is not None and self.method != self.GLB7DoF:
            _, pcd_part = self.pcdloss(part_verts_canonical, self.pcd1.unsqueeze(0))
        else:
            pcd_part = None

        if self.method == self.BOTH:
            _, constraint_loss = self.pcdloss(self.pcd.unsqueeze(0), self.pcd1.unsqueeze(0))
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

        return [pcd_loss, pcd_base, pcd_part], seg_loss, depth_loss, all_seg, all_depth2d[..., 0]


class Constraints(nn.Module):
    """
    Hand pose constraints
    """

    def __init__(self, cuda_id):
        super(Constraints, self).__init__()
        self.cuda_device = cuda_id
        self.thetaLimits()

    def thetaLimits(self):
        MINBOUND = -5.
        MAXBOUND = 5.
        self.validThetaIDs = torch.IntTensor([0, 1, 2, 3, 4, 5, 6, 8, 11, 13, 14, 15, 17, 20, 21, 22, 23, 25, 26, 29, 30, 31, 32, 33, 35, 38, 39, 40, 41, 42, 44, 46, 47]).long().cuda()
        #7,9,10,12,16,18,19,24,27,28,34,36,37,43,45 15个
        #48-15=33个
        # self.invalidThetaIDs = np.array([7, 9, 10, 12, 16, 18, 19, 24,
        #                                25, 27, 28, 34, 36, 37, 39, 43, 45], dtype=np.int32)
        invalidThetaIDsList = []
        for i in range(48):
            if i not in self.validThetaIDs:
                invalidThetaIDsList.append(i)
        self.invalidThetaIDs = np.array(invalidThetaIDsList)

        self.minThetaVals = nn.Parameter(torch.FloatTensor([MINBOUND, MINBOUND, MINBOUND,  # global rot
                                      0, -0.15, 0.1, -0.3, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # index
                                      MINBOUND, -0.15, 0.1, -0.5, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # middle
                                      -1.5, -0.15, -0.1, MINBOUND, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # pinky
                                      -0.5, -0.25, 0.1, -0.4, MINBOUND, -0.0, MINBOUND, MINBOUND, 0,  # ring
                                               MINBOUND, -0.83, -0.0, -0.15, MINBOUND, 0, MINBOUND, -0.5, -1.57, ])) # thumb

        self.maxThetaVals = nn.Parameter(torch.FloatTensor([MAXBOUND, MAXBOUND, MAXBOUND,  # global
                                      0.45, 0.2, 1.8, 0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # index
                                      MAXBOUND, 0.15, 2.0, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # middle
                                      -0.2, 0.15, 1.6, MAXBOUND, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # pinky
                                      -0.4, 0.10, 1.6, -0.2, MAXBOUND, 2.0, MAXBOUND, MAXBOUND, 1.25,  # ring
                                               MAXBOUND, 0.66, 0.5, 1.6, MAXBOUND, 0.5, MAXBOUND, 0, 1.08]))  # thumb

        self.fullThetaMat = np.zeros(
            (48, len(self.validThetaIDs)), dtype=np.float32)  # 48x25
        for i in range(len(self.validThetaIDs)):
            self.fullThetaMat[self.validThetaIDs[i], i] = 1.0
        self.minThetaVals.requires_grad = False
        self.maxThetaVals.requires_grad = False
        self.validThetaIDs.requires_grad = False


    def forward(self, theta, isValidTheta=False):
        '''
        get constraints on the joint angles when input is theta vector itself (first 3 elems are NOT global rot)
        :param theta: Nx45 tensor if isValidTheta is False and Nx25 if isValidTheta is True
        :param isValidTheta:
        :return:
        '''
        if not isValidTheta:
            assert (theta.shape)[-1] == 45
            validTheta = theta[self.validThetaIDs[3:] - 3]
            #使用的fullpose只有0, 1, 2, 3, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 22, 23, 26,
            #27, 28, 29, 30, 32, 35, 36, 37, 38, 39, 41, 43, 44
        else:
            assert (theta.shape)[-1] == len(self.validThetaIDs[3:])
            validTheta = theta

        phyConstMax = (torch.maximum(
            self.minThetaVals[self.validThetaIDs[3:]] - validTheta, torch.zeros(30).to(self.cuda_device)))
        phyConstMin = (torch.maximum(
            validTheta - self.maxThetaVals[self.validThetaIDs[3:]], torch.zeros(30).to(self.cuda_device)))

        return phyConstMin, phyConstMax


class HandObj(nn.Module):
    """
    SingleFrame Hand Pose Refine
    theta*30, beta, t, gt_seg, gt_dpt, gt_pcd, phy_cons, ... -> losses
    hand_v = mano(theta, beta) + T
    pcd_loss = EMDloss(hand_v, gt_pcd)
    rendered_seg, rendered_dpt = SoftRas(projected_faces)
    seg_loss = ~HandMask * l1_loss(rendered_seg, gt_seg)
    dpt_loss = ~HandMask * l2_loss(rendered_dpt, gt_dpt)
    """

    def __init__(self, batch_size, ncomps, poseCoeff, trans, beta, kps2d, vis,
                 handseg, objmask, handdpt, handpcd, camMat, gpu, crop=(0, 1080, 0, 1920), size=1920, w=1920, h=1080, pca = True, flat_bool=True, vis_bool = False):
        super(HandObj, self).__init__()
        self.visbool = vis_bool
        self.batch_size = batch_size
        self.theta = nn.Parameter(torch.FloatTensor(
            poseCoeff).expand(batch_size, poseCoeff.shape[0]))
        self.beta = nn.Parameter(torch.FloatTensor(
            beta).expand(batch_size, beta.shape[0]))
        self.trans = nn.Parameter(torch.FloatTensor(trans))

        self.mano_layer = ManoLayer(
            mano_root='manopth/mano/models', use_pca=pca, ncomps=ncomps, flat_hand_mean=flat_bool)
        self.pcdloss = EMDLoss()
        self.cdloss = ChamferDistance()

        self.cam2pix = projCamera(camMat)

        self.poseConstraint = Constraints(gpu)

        self.seg = nn.Parameter(torch.FloatTensor(handseg))

        self.dpt = nn.Parameter(torch.FloatTensor(handdpt))
        self.msk = nn.Parameter(torch.FloatTensor(objmask))
        self.pcd = nn.Parameter(torch.FloatTensor(handpcd))
        self.kps2d = nn.Parameter(torch.FloatTensor(kps2d))
        self.vis = nn.Parameter(torch.FloatTensor(vis))
        self.seg.requires_grad = False
        self.msk.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False
        self.pcd.requires_grad = False
        self.kps2d.requires_grad = False
        self.vis.requires_grad = False
        # self.beta.requires_grad = False

        self.imsize = size
        self.w = w
        self.h = h
        self.crop = crop
        self.center = nn.Parameter(torch.FloatTensor([h//2, w//2]))
        self.center.requires_grad = False
        self.camMat = nn.Parameter(torch.FloatTensor(camMat))
        self.camMat.requires_grad = False

    
    def projCamera(self, mesh):
        mesh = mesh.permute(0, 2, 1)
        mesh = torch.einsum('ik, bkj -> bij', self.camMat, mesh)
        mesh = mesh.permute(0, 2, 1)
        ret1 = mesh[..., 0] / mesh[..., 2]
        ret2 = mesh[..., 1] / mesh[..., 2]
        ret3 = mesh[..., 2]
        ret = torch.stack([ret1, ret2, ret3], dim=-1)
        return ret
    
    def projKps(self, kps):
        '''
        # 示例: 从相机坐标系(3D坐标)转到像素坐标系(像素坐标)
        ans = pcd.transpose()
        ans = np.dot(camMat, ans)
        ans = ans.transpose()
        ans1 = (ans[:, 0] / ans[:, 2]).reshape(-1, 1)
        ans2 = (ans[:, 1] / ans[:, 2]).reshape(-1, 1)
        print(np.concatenate((ans1, ans2), axis=1))
        '''
        kps = kps.permute(0, 2, 1)
        kps = torch.einsum('ik, bkj -> bij', self.camMat, kps)
        kps = kps.permute(0, 2, 1)
        ret1 = kps[..., 0] / kps[..., 2]
        ret2 = kps[..., 1] / kps[..., 2]
        ret3 = kps[..., 2]
        ret = torch.stack([ret1, ret2, ret3], dim=-1)
        return ret
    
    def cal_seg_loss(self,rendered_seg, gt_seg):
        assert rendered_seg.shape == gt_seg.shape, 'shape should be the same'
        return 1 - torch.sum(rendered_seg * gt_seg) / torch.sum(rendered_seg + gt_seg - rendered_seg * gt_seg)


    def forward(self):
        hand_verts, hand_joints, full_pose = self.mano_layer(self.theta, self.beta)

        # hand_fullpose = self.mano_layer.th_comps
        hand_faces_index = self.mano_layer.th_faces.detach().cpu().numpy()
        hand_faces = hand_verts[0, hand_faces_index]
        # print(hand_faces_index.shape)
        # print(hand_faces.shape)
        transformed_verts = hand_verts/1000.0 + self.trans  # torch.Size([1, 778, 3])
        transformed_joints = hand_joints/1000.0 + self.trans  # torch.Size([1, 21, 3])
        transformed_faces = hand_faces/1000.0 + self.trans  # torch.Size([1538, 3, 3])


        # projected_faces = self.cam2pix(transformed_faces)
        projected_faces = self.projCamera(transformed_faces)
        projected_faces = projected_faces.unsqueeze(0)  # 像素系mesh各点的像素坐标+深度值
        # projected_joints = torch.stack((self.Kps3Dto2D(transformed_joints)[0, :, 0]*self.camMat[0][0]+self.center[0],
        #                                self.Kps3Dto2D(transformed_joints)[0, :, 1]*self.camMat[1][1]+self.center[1]), dim=1)
        projected_joints = self.projKps(transformed_joints)[0, :, :2]  # 像素系21个关键点的坐标
        
        
        # 可视化
        p = o3d.geometry.PointCloud()
        pts_ = transformed_faces.reshape(-1, 3).detach().cpu().numpy()
        pts_ = np.concatenate((pts_, transformed_joints[0].detach().cpu().numpy()), axis=0)
        p.points = o3d.utility.Vector3dVector(pts_)
        p.paint_uniform_color([0, 0.651, 0.929])
        
        p_gt = o3d.geometry.PointCloud()
        pts_gt = self.pcd.reshape(-1, 3).detach().cpu().numpy()
        p_gt.points = o3d.utility.Vector3dVector(pts_gt)
        p_gt.paint_uniform_color([0.651, 0.929,0])
        # print(p.points)
        # o3d.io.write_point_cloud("./ex.ply", p)
        p_gt += p
        # o3d.io.write_point_cloud("./ex.ply", p_gt)
        
        

        # print("projected_faces", projected_faces[0, :, :, :])
        # print("projected_joints", projected_joints)
        '''
        img = cv2.imread("/home/jiangche/HOI4D/subject2_h1_0/align_image/000299.png")
        for i in range(projected_faces.shape[1]):
            for j in range(projected_faces.shape[2]):
                cv2.circle(img, center=(int(projected_faces[0, i, j, 0]), int(projected_faces[0, i, j, 1])), radius=1, color=[0, 255, 0], thickness=-1)
        for i in range(projected_faces.shape[1]):
            x0 = int(projected_faces[0, i, 0, 0])
            y0 = int(projected_faces[0, i, 0, 1])
            x1 = int(projected_faces[0, i, 1, 0])
            y1 = int(projected_faces[0, i, 1, 1])
            x2 = int(projected_faces[0, i, 1, 0])
            y2 = int(projected_faces[0, i, 1, 1])
            cv2.line(img, (x0, y0), (x1, y1), (0, 0, 255))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))
            cv2.line(img, (x2, y2), (x0, y0), (0, 0, 255))
        for i in range(projected_joints.shape[0]):
            cv2.circle(img, center=(int(projected_joints[i, 0]), int(projected_joints[i, 1])), radius=3, color=[255, 255, 255], thickness=-1)
        cv2.imwrite("./ex.png", img)
        print("visualization ok!")
        '''

        depth = torch.stack([projected_faces[..., 2]]*3, dim=-1)
        
        #render_result = srf.soft_rasterize(projected_faces, depth,
        #                                   self.imsize, texture_type='vertex', near=0.5)
        
        # assert int(self.camMat[0, 0]) == int(self.camMat[1, 1])  # 要求srf.soft_rasterize输入的图像是正方形
        assert self.camMat[0, 0] > self.camMat[0, 2]
        assert self.camMat[0, 0] > self.camMat[1, 2]
        alpha = 1  # alpha是整数，其设置理论上不改变render结果，只是为了防止render的图片分辨率不够大，但现在结果随着alpha的增大变“胖”, 是由于SoftRas一些默认参数导致的
        f = int(self.camMat[0, 0]) * alpha
        input_faces = transformed_faces.clone()
        input_faces[..., 0] = transformed_faces[..., 0] / transformed_faces[..., 2] / alpha
        input_faces[..., 1] = - transformed_faces[..., 1] / transformed_faces[..., 2] / alpha * self.camMat[1, 1] / self.camMat[0, 0]
        '''
        # change the depth scale to [1, 100]
        depth_mn = input_faces[:, :, 2].min()
        depth_mx = input_faces[:, :, 2].max()
        print(depth_mn, depth_mx)
        input_faces[:, :, 2] = (input_faces[:, :, 2] - depth_mn) / (depth_mx - depth_mn) * (100 - 1) + 1
        print(input_faces)
        '''
        '''
        wr = open("./debug_ex1.out", "w")
        for i in range(depth.shape[1]):
            for j in range(depth.shape[2]):
                wr.write("{} ".format(str(depth[0, i, j, 0])))
            wr.write("\n")
        wr.close()
        # depth[:, :, :, :] = 0.5
        '''
        # background_color = [0, 0, 0]
        render_result = srf.soft_rasterize(input_faces.unsqueeze(0), depth,
                                           f * 2, background_color=[0, 0, 0], texture_type='vertex', near=0.1, far=101, sigma_val=1e-5, gamma_val=1e-5, eps=1e-4, dist_eps=1e-5)
        

        render_result = render_result.squeeze(0).permute((1, 2, 0))
        start_idx = f - int(self.camMat[0, 2])
        start_idy = f - int(self.camMat[1, 2])

        rendered_depth = render_result[start_idy: start_idy + self.h, start_idx: start_idx + self.w, 0]
        rendered_seg = render_result[start_idy: start_idy + self.h, start_idx: start_idx + self.w, 3]

        
        # 可视化
        dep = render_result[start_idy: start_idy + self.h, start_idx: start_idx + self.w].detach().cpu().numpy()
        alpha = dep[:, :, 3]
        image = (255 * dep).astype(np.uint8)
        #print(dep[0, 0], dep[550, 858], dep[509, 863], dep[566, 792])
        img = image[:, :, :3]
        img[alpha < 0.5] = 0
        if self.visbool:
            cv2.imwrite("./ex1.png", image[:, :, :3])
        p = o3d.geometry.PointCloud()
        pts_depth_ = dep[alpha > 0.5][:, 0]
        pts_y_, pts_x_ = np.where(alpha > 0.5)
        pts_ = np.stack((pts_x_ * pts_depth_, pts_y_ * pts_depth_, pts_depth_), axis=1)
        camMat = self.camMat.detach().cpu().numpy()
        pts_ = np.dot(np.linalg.inv(camMat), pts_.transpose()).transpose()
        # print("pts_", pts_.shape, pts_)
        p.points = o3d.utility.Vector3dVector(pts_)
        p.paint_uniform_color([0.651, 0, 0.929])
        p_gt += p
        # o3d.io.write_point_cloud("./ex.ply", p_gt)
        

        max_idx = min(2*f-1, self.w)
        max_idy = min(2*f-1, self.h)

        # 不训练
        # results = {
        #     'seg': rendered_seg,
        #     'dep': rendered_depth,
        #     '2Djoints': projected_joints
        # }
        # return 0, 0, 0, 0, 0, 0, 0, results
        


        constMin, constMax = self.poseConstraint(full_pose[0][3:])
        constMin_loss = torch.norm(constMin)
        constMax_loss = torch.norm(constMax)
        invalidTheta_loss = torch.norm(full_pose[0][self.poseConstraint.invalidThetaIDs])
        
        # print(transformed_verts.contiguous().shape, self.pcd.unsqueeze(0).shape)  # torch.Size([1, 778, 3]), torch.Size([1, 5693, 3])
        # 问题：EMD loss要求两个点云的点数相同，这里输入的点数不同！
        pcd_loss = self.pcdloss( transformed_verts.contiguous(), self.pcd.unsqueeze(0) )
        # dist1, dist2, _, _ = self.cdloss(transformed_verts.contiguous(), self.pcd.unsqueeze(0))


        # pcd_loss = torch.mean(dist2)
        # print(transformed_verts.contiguous().shape, self.pcd.unsqueeze(0).shape)
        # exit(0)
        
        # 问题：seg_loss和dpt_loss无法被训练！
        # print(np.unique(self.msk.detach().cpu().numpy()))
        # print(rendered_seg.shape, self.seg.shape, self.msk.shape)

        rendered_seg_clip = (rendered_seg > 0.5).int()
        if self.visbool:
            img = (rendered_seg_clip - self.seg[:max_idy, :max_idx, 0]) * self.msk[:max_idy, :max_idx, 0]
            # img = rendered_seg_clip - self.seg[:max_idy, :max_idx, 0]
            img = (255*img.detach().cpu().numpy()).astype(np.uint8)

            cv2.imwrite("./ex2.png", img)
        

        # seg_loss = torch.mean(torch.abs(rendered_seg_clip - self.seg[:max_idy, :max_idx, 0]) * self.msk[:max_idy, :max_idx, 0])
        seg_loss = self.cal_seg_loss(rendered_seg* self.msk[:max_idy, :max_idx, 0], self.seg[:max_idy, :max_idx, 0])
        # dpt_loss = torch.mean(torch.abs(rendered_depth ))
        # seg_loss = 0
        # dpt_loss = torch.mean(torch.abs(
        #     rendered_depth - self.dpt[:max_idy, :max_idx]) * self.dpt[:max_idy, :max_idx])
        r_depth = rendered_depth[:max_idy, :max_idx]
        gt_depth = self.dpt[:max_idy, :max_idx]
        valid_flag = (rendered_seg > 0.5) & (gt_depth > 0)
        r_depth_valid = r_depth[valid_flag]
        gt_depth_valid = gt_depth[valid_flag]
        dpt_loss = torch.mean(torch.abs(r_depth_valid - gt_depth_valid))

        # print(projected_joints, self.kps2d, self.vis)
        # print(projected_joints.shape, self.kps2d.shape, self.vis.shape)
        # print((projected_joints - self.kps2d)*self.vis)
        # print(projected_joints - self.kps2d)
        kps2d_loss = torch.mean(torch.square(
            (projected_joints - self.kps2d) * self.vis))
        # print((projected_joints - self.kps2d)*self.vis)
        
        # wrist_depth_loss = torch.abs(gt_depth[self.kps2d[0][1].int(), self.kps2d[0][0].int()] - r_depth[self.kps2d[0][1].int(), self.kps2d[0][0].int()]) \
        #     * (gt_depth[self.kps2d[0][1].int(), self.kps2d[0][0].int()] > 0)
        wrist_depth_loss = [0]

        results = {
            'seg': rendered_seg,
            'dep': rendered_depth,
            '2Djoints': projected_joints,
            'pointcloud': p_gt
        }

        return pcd_loss, seg_loss, dpt_loss, kps2d_loss, constMin_loss, constMax_loss, \
            invalidTheta_loss, wrist_depth_loss, results

