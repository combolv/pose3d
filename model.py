import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

import numpy as np
import json
import argparse
import cv2
import open3d as o3d

import soft_renderer.functional as srf
from emd import EMDLoss


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
    x = fx * x / z
    y = - fy * y / z
    """
    def __init__(self, camMat):
        super(projCamera, self).__init__()
        self.xscale = camMat[0, 0] / 1000
        self.yscale = camMat[1, 1] / 1000

    def forward(self, mesh):
        ret1 = self.xscale * mesh[..., 0] / mesh[..., 2]
        # SoftRas uses a left-hand coordinate system
        ret2 = - self.yscale * mesh[..., 1] / mesh[..., 2]
        ret3 = mesh[..., 2]
        return torch.stack([ret1, ret2, ret3], dim=-1)


class PlainObj(nn.Module):
    """
    Single Rigid Object Pose Refine
    R, T, CAD_model, gt_seg, gt_dpt, gt_pcd, ... -> losses, rendered_seg, rendered_
    transformed_v = R v + T
    pcd_loss = EMDloss(transformed_v, gt_pcd)
    rendered_seg, rendered_dpt = SoftRas(projected_faces)
    seg_loss = ~HandMask * l1_loss(rendered_seg, gt_seg)
    dpt_loss = ~HandMask * l2_loss(rendered_dpt, gt_dpt)
    """
    def __init__(self, rotvec, trans, vertices, triangles, objseg, handMask, objdpt, pcd, camMat, crop=(0,1080,0,1920), size=1920):
        super(PlainObj, self).__init__()
        self.rotvec = nn.Parameter(torch.FloatTensor(rotvec))
        self.trans = nn.Parameter(torch.FloatTensor(trans))

        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))
        self.faces.requires_grad = False
        self.vertices.requires_grad = False

        self.vec2mat = invRodrigues()
        self.cam2pix = projCamera(camMat)
        self.pcdloss = EMDLoss()

        self.seg = nn.Parameter(torch.FloatTensor(objseg)[..., 0])
        self.dpt = nn.Parameter(torch.stack([torch.FloatTensor(objdpt)]*3, dim=-1))
        self.msk = nn.Parameter(torch.FloatTensor(handMask))
        self.pcd = nn.Parameter(torch.FloatTensor(pcd))
        self.seg.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False
        self.pcd.requires_grad = False

        self.imsize = size
        self.crop = crop

    def forward(self):
        R_inv = self.vec2mat(self.rotvec)
        transformed_faces = torch.matmul(self.faces, R_inv) + self.trans
        transformed_vertices = torch.matmul(self.vertices, R_inv) + self.trans
        projected_faces = self.cam2pix(transformed_faces)
        projected_faces = projected_faces.unsqueeze(0)
        depth = torch.stack([projected_faces[..., 2]]*3, dim=-1)

        render_result = srf.soft_rasterize(projected_faces, depth,
                                        self.imsize, texture_type='vertex', near=0.5)
        render_result = render_result.squeeze(0).permute((1, 2, 0))
        rendered_depth = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], :3]
        rendered_seg = render_result[self.crop[0]:self.crop[1], self.crop[2]:self.crop[3], 3]

        pcd_loss = self.pcdloss(transformed_vertices.unsqueeze(0), self.pcd.unsqueeze(0))
        seg_loss = torch.mean(torch.abs(rendered_seg - self.seg) * self.msk[:, :, 0])

        dpt_loss = torch.mean(torch.square(rendered_depth - self.dpt) * self.msk)

        return pcd_loss, seg_loss, dpt_loss, rendered_seg, rendered_depth


class BatchObj(nn.Module):
    """
    Support multi-frame inputs
    Use project_faces = torch.stack([project_faces])
    Not implemented yet
    """

    def __init__(self, rotvec, trans, objmesh, objseg, handMask, objdpt, pcd, camMat, crop=(0, 1080, 0, 1920),
                 size=1920):
        super(BatchObj, self).__init__()
        self.rotvec = nn.Parameter(torch.FloatTensor(rotvec))
        self.trans = nn.Parameter(torch.FloatTensor(trans))

        vertices = np.asarray(objmesh.vertices)

        triangles = np.asarray(objmesh.triangles)
        self.faces = nn.Parameter(torch.FloatTensor(vertices[triangles]))
        self.vertices = nn.Parameter(torch.FloatTensor(vertices))
        self.faces.requires_grad = False
        self.vertices.requires_grad = False

        self.vec2mat = invRodrigues()
        self.cam2pix = projCamera(camMat)
        self.pcdloss = EMDLoss()

        self.seg = nn.Parameter(torch.FloatTensor(objseg)[..., 0])
        self.dpt = nn.Parameter(torch.stack([torch.FloatTensor(objdpt)] * 3, dim=-1))
        self.msk = nn.Parameter(torch.FloatTensor(handMask))
        self.pcd = nn.Parameter(torch.FloatTensor(pcd))
        self.seg.requires_grad = False
        self.dpt.requires_grad = False
        self.msk.requires_grad = False
        self.pcd.requires_grad = False

        self.imsize = size
        self.crop = crop

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


class ArtObj(nn.Module):
    """
    Support Articulated Object inputs
    Not implemented yet
    """
    def __init__(self, paras, objmesh, objseg, handMask, objdpt, pcd, camMat, crop=(0,1080,0,1920), size=1920):
        super(ArtObj, self).__init__()
        trans, rot, part_trans, part_rot, link_axis = paras
        global_mesh, part_mesh = objmesh

        raise NotImplementedError


class HandObj(nn.Module):
    """
    Support hands
    Not implemented yet
    """
    def __int__(self):
        super(HandObj, self).__init__()
        raise NotImplementedError


"""
def single_frame_pipeline(cud, cam, dep, msk, jsn, obj, rot_in=None, t_in=None, epoch=25):
    cuda_device = cud
    mesh = o3d.io.read_triangle_mesh(obj)
    depth2d = cv2.imread(dep, cv2.IMREAD_UNCHANGED)
    camMat = np.load(cam)
    anno_path = jsn
    mask_path = msk
    rot, trans = read_rt(anno_path)
    if rot_in is not None:
        print(rot, trans, rot_in, t_in)
        rot = rot_in
        trans = t_in
    x1, x2, y1, y2, c12, c02 = read_mask2bbox(mask_path)

    crop_list = [x1, x2, y1, y2]

    depth2d = np.array(depth2d, dtype=np.float32) / 1000
    obj_mask, hand_mask = read_mask(mask_path, dscale=1, crop=crop_list)

    large_mask, _ = read_mask(mask_path, dscale=1)

    depth3d = o3d.geometry.Image(depth2d * large_mask[..., 0])
    depth2d = depth2d[x1:x2, y1:y2]

    # simplify the CAD model
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
    mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                               contraction=o3d.geometry.SimplificationContraction.Average)


    # load point cloud from depth, NOTE: If scene reconstruction is here, it will be better.
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0, 0], camMat[1, 1], camMat[0, 2], camMat[1, 2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d, intrinsics, stride=2)
    # o3d.visualization.draw_geometries([pcd])
    pcd = np.asarray(pcd.points)

    crop_list = [x1 - int(np.round(camMat[1, 2])) + 960, x2 - int(np.round(camMat[1, 2])) + 960,
                 y1 - int(np.round(camMat[0, 2])) + 960, y2 - int(np.round(camMat[0, 2])) + 960]

    model = PlainObj(rot, trans, mesh_smp, obj_mask, hand_mask, depth2d, pcd, camMat, crop_list)
    model.to(cuda_device)


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in tqdm(range(epoch)):
        a, b, c, _, _ = model()
        loss = 0.01 * a + 10 * b + 2 * c
        optimizer.zero_grad()
        loss.backward()
        # input()
        optimizer.step()

    # a, b, c, rendered_seg, rendered_depth = model()
    rot_vec = model.rotvec.data.detach().cpu().numpy()
    trans = model.trans.data.detach().cpu().numpy()
    # print(a.item(), b.item(), c.item(), rot_vec, trans)
    return rot_vec, trans



if __name__ == '__main__':
    from loadfile import read_mask2bbox, read_rt, read_mask, read_total_json
    cuda_device = 'cuda:0'

    mesh, depth2d, camMat, anno_path, mask_path = None, None, None, None, None
    for cam, dep, msk, jsn, obj in read_total_json('total.json'):
        mesh = o3d.io.read_triangle_mesh(obj)
        depth2d = cv2.imread(dep, cv2.IMREAD_UNCHANGED)
        camMat = np.load(cam)
        anno_path = jsn
        mask_path = msk
        break

    rot, trans = read_rt(anno_path)
    x1, x2, y1, y2, c12, c02 = read_mask2bbox(mask_path)

    crop_list = [x1, x2, y1, y2]

    depth2d = np.array(depth2d, dtype=np.float32) / 1000
    obj_mask, hand_mask = read_mask(mask_path, dscale=1, crop=crop_list)

    large_mask, _ = read_mask(mask_path, dscale=1)

    depth3d = o3d.geometry.Image(depth2d * large_mask[..., 0])
    depth2d = depth2d[x1:x2, y1:y2]


    # simplify the CAD model
    voxel_size = max(mesh.get_max_bound()-mesh.get_min_bound()) / 32
    mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    # load point cloud from depth, NOTE: If scene reconstruction is here, it will be better.
    intrinsics = o3d.camera.PinholeCameraIntrinsic(1920, 1080, camMat[0,0], camMat[1,1], camMat[0,2], camMat[1,2])
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth3d, intrinsics, stride=2)
    # o3d.visualization.draw_geometries([pcd])
    pcd = np.asarray(pcd.points)

    crop_list = [x1 - int(np.round(camMat[1, 2])) + 960, x2 - int(np.round(camMat[1, 2])) + 960,
                 y1 - int(np.round(camMat[0, 2])) + 960, y2 - int(np.round(camMat[0, 2])) + 960]

    model = PlainObj(rot, trans, mesh_smp, obj_mask, hand_mask, depth2d, pcd, camMat, crop_list)
    model.to(cuda_device)

    a, b, c, rendered_seg, rendered_depth = model()

    print(a.item(), b.item(), c.item())
    # cv2.imwrite('output4.png', ((200 * rendered_seg).detach().cpu().numpy()).astype(np.uint8))
    #
    # cv2.imwrite('output5.png', (1600 * (rendered_depth - 0.85).detach().cpu().numpy()).astype(np.uint8))

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for _ in tqdm(range(200)):
        a, b, c, _, _ = model()
        loss = 0.01 * a + 10 * b + 2 * c
        optimizer.zero_grad()
        loss.backward()
        print(model.rotvec.data)
        print(model.rotvec.grad)
        # input()
        optimizer.step()

    a, b, c, rendered_seg, rendered_depth = model()
    rot_vec = model.rotvec.data.detach().cpu().numpy()
    trans = model.trans.data.detach().cpu().numpy()
    print(a.item(), b.item(), c.item(), rot_vec, trans)


    # cv2.imwrite('output.png', ((200 * rendered_seg).detach().cpu().numpy()).astype(np.uint8))
    #
    # cv2.imwrite('output2.png', (1600*(depth2d-0.85)).astype(np.uint8))
    #
    # cv2.imwrite('output3.png', (1600*(rendered_depth-0.85).detach().cpu().numpy()).astype(np.uint8))
"""