import json
import os
import open3d as o3d
import torch
import numpy as np


class SingleAxisObject:
    def __init__(self, base_verts, base_idx, part_verts, part_idx,
                 part_align_verts, part_align_idx,
                 se3, axis_orig, axis_dir, degree_max, degree_min):
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


def simplify_mesh(mesh):
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
    mesh_smp =  mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                            contraction=o3d.geometry.SimplificationContraction.Average)
    return np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles)


def pcd_center(pcd):
    return 0.5 * (pcd.get_max_bound() + pcd.get_min_bound())


def mesh2pcd(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices)), o3d.utility.Vector3iVector(triangles)


def registration(pcd1, pcd2): # source == mesh1, target == mesh2

    def fpfh_compute(pcd):
        radius_normal = 0.01
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = 0.025
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid
                                                                   (radius=radius_feature, max_nn=50))
        return pcd_fpfh


    def execute_fast_global_registration(source, target, source_fpfh, target_fpfh):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, 0.5)
        return result

    source, target = pcd1, pcd2
    # source = mesh2pcd(mesh1)
    # target = mesh2pcd(mesh2)

    source_fpfh = fpfh_compute(source)
    target_fpfh = fpfh_compute(target)

    result_fast = execute_fast_global_registration(source, target,
                                                   source_fpfh, target_fpfh)

    return result_fast.transformation


def simplify_CAD_model_and_get_rel_pose(base_path, save_path, base_name = "body", part_name = "handle", json0_path = None,
                                        size_rescale = "basemode"):
    dim, dim_part = None, None
    if json0_path is not None:
        assert size_rescale == "basemode"
        with open(json0_path, "r") as f:
            cont = json.load(f)
            anno = cont["dataList"][0]["dimensions"]
            dim = np.array([anno['length'], anno['width'], anno['height']], dtype=np.float32)
            anno = cont["dataList"][1]["dimensions"]
            dim_part = np.array([anno['length'], anno['width'], anno['height']], dtype=np.float32)
    result_json = os.path.join(base_path, "result.json")
    v2json = os.path.join(base_path, "mobility_v2.json")
    base_obj_name, part_obj_name = None, None
    with open(result_json, "r") as f:
        cont = json.load(f)
        parts = cont[0]["children"]
        for part in parts:
            if part["name"] == base_name:
                base_obj_name = part["objs"][0]
            elif part["name"] == part_name:
                part_obj_name = part["objs"][0]
            else:
                raise AssertionError("No part named " + part["name"])
        assert base_obj_name is not None
        assert part_obj_name is not None
    origin, direction, limit_a, limit_b = None, None, None, None
    with open(v2json, "r") as f:
        cont = json.load(f)
        for part in cont:
            if part["name"] == part_name:
                joint_info = part["jointData"]
                origin = joint_info["axis"]["origin"]
                direction = joint_info["axis"]["direction"]
                limit_a = joint_info["limit"]["a"]
                limit_b = joint_info["limit"]["b"]
                if limit_a <= limit_b:
                    limit_a, limit_b = limit_b, limit_a
                break
        assert origin is not None
    base_obj_file = os.path.join(base_path, "objs", base_obj_name + ".obj")
    assert os.path.exists(base_obj_file)
    base_obj_align_file = os.path.join(base_path, "objs", base_obj_name + "-align.obj")
    assert os.path.exists(base_obj_align_file)
    part_obj_file = os.path.join(base_path, "objs", part_obj_name + ".obj")
    assert os.path.exists(part_obj_file)
    part_obj_align_file = os.path.join(base_path, "objs", part_obj_name + "-align.obj")
    assert os.path.exists(part_obj_align_file)

    base_obj = o3d.io.read_triangle_mesh(base_obj_file)
    part_obj = o3d.io.read_triangle_mesh(part_obj_file)
    part_align_obj = o3d.io.read_triangle_mesh(part_obj_align_file)

    base_pcd, base_3i = mesh2pcd(base_obj)
    part_pcd, part_3i = mesh2pcd(part_obj)
    part_align_pcd, part_align_3i = mesh2pcd(part_align_obj)

    base_center = pcd_center(base_pcd)
    base_pcd = base_pcd.translate( - base_center )
    part_pcd = part_pcd.translate( - base_center )
    origin -= base_center
    part_align_pcd = part_align_pcd.translate( - pcd_center(part_align_pcd) )

    if dim is not None:
        base_pcd_size = base_pcd.get_max_bound() - base_pcd.get_min_bound()
        part_pcd_size = part_align_pcd.get_max_bound() - part_align_pcd.get_min_bound()
        scalars = dim / base_pcd_size
        scalar_mul = np.median(scalars)
        print("桶体标注：", dim, "\n把手标注：", dim_part,
              "\n桶体模型外接框大小：", base_pcd_size, "\n把手模型外接框大小：", part_pcd_size)#, scalar_mul)
        input()
        raise NotImplementedError

    # icp registration
    rel_se3_mat = registration(part_align_pcd, part_pcd)

    print(len(base_pcd.points), len(part_pcd.points), len(part_align_pcd.points))
    base_mesh_rebuild = o3d.geometry.TriangleMesh(
        base_pcd.points,
        base_3i
    )
    part_mesh_rebuild = o3d.geometry.TriangleMesh(
        part_pcd.points,
        part_3i
    )
    part_align_rebuild = o3d.geometry.TriangleMesh(
        part_align_pcd.points,
        part_align_3i
    )
    base_vertices, base_triangles = simplify_mesh(base_mesh_rebuild)
    part_vertices, part_triangles = simplify_mesh(part_mesh_rebuild)
    part_align_vertices, part_align_triangles = simplify_mesh(part_align_rebuild)

    print(len(base_triangles), len(part_triangles), len(part_align_triangles))

    print(np.max(base_vertices.reshape((-1, 3)), axis=0) - np.min(base_vertices.reshape((-1, 3)), axis=0),
          np.max(part_align_vertices.reshape((-1, 3)), axis=0) - np.min(part_align_vertices.reshape((-1, 3)), axis=0))
    # input()
    # raise RuntimeWarning
    fin_obj = SingleAxisObject(base_vertices, base_triangles, part_vertices,
                               part_triangles, part_align_vertices, part_align_triangles,
                               rel_se3_mat, origin, direction, limit_a, limit_b)
    torch.save(fin_obj, os.path.join(save_path))

    # o3d.visualization.draw_geometries([base_mesh_rebuild, part_mesh_rebuild])
    # o3d.visualization.draw_geometries([base_mesh_rebuild, part_align_rebuild])


def load_test():
    a = torch.load("/nas/HOI4D_ObjPose_Cache/bucket_test/009.pt")


if __name__ == "__main__":
    # load_test()
    simplify_CAD_model_and_get_rel_pose("/mnt/8T/HOI4D_CAD_Model/part_annotations/水桶/003/",
                                        "/nas/HOI4D_ObjPose_Cache/bucket_test/003.pt")
                                        # json0_path="../bucket3/0.json")