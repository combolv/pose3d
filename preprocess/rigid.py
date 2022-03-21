import os
import open3d as o3d
import numpy as np


def test():
    orig_path = "/mnt/8T/HOI4D_CAD_Model/models_watertight_scale/"
    save_path = '/nas/HOI4D_ObjPose_Cache/cad'
    for s in os.listdir(orig_path):
        out_path = os.path.join(save_path, s)
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        now_path = os.path.join(orig_path, s)
        for idx in range(80):
            obj_file_name = str(idx).zfill(3) + ".obj"
            obj_path = os.path.join(now_path, obj_file_name)
            if not os.path.exists(obj_path):
                print(obj_path, "failed")
                continue
            else:
                mesh = o3d.io.read_triangle_mesh(obj_path)

                # simplify the CAD model
                voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 32
                mesh_smp = mesh.simplify_vertex_clustering(voxel_size=voxel_size,
                                                           contraction=o3d.geometry.SimplificationContraction.Average)
            vertices = np.asarray(mesh_smp.vertices)
            triangles = np.asarray(mesh_smp.triangles)
            print(len(vertices), "succ")
            # np.save(os.path.join(out_path, obj_file_name + "v.npy"), vertices)
            # np.save(os.path.join(out_path, obj_file_name + "t.npy"), triangles)

test()