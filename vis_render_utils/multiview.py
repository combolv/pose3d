import open3d as o3d
import numpy as np
import json
from scipy.spatial.transform import Rotation as Rt
from copy import deepcopy

json_proto = {
	"class_name" : "PinholeCameraParameters",
	"extrinsic" :
	[
		[0.9503879347691897,
		-0.12228840479714526,
		0.28602153677183806,
		0.0],
		[0.095412504779002483,
		0.98976322388070337,
		0.10613771518679699,
		0.0],
		[-0.29607301021361471,
		-0.073581972693359823,
		0.95232686926160104,
		0.0],
		[-1.8429529140825325,
		-1.2214317950963878,
		0.6428005992092698,
		1.0]
	],
	"intrinsic" :
	{
		"height" : 1080,
		"intrinsic_matrix" : None,
		"width" : 1920
	},
	"version_major" : 1,
	"version_minor" : 0
}

def generate_json_file(cam_intrinsic_mat = None):
    if cam_intrinsic_mat is None:
        cam_intrinsic_mat = np.load("/home/yunze/pose3d_lkb/1.npy")
    json_list = deepcopy(json_proto)
    json_list["intrinsic"]["intrinsic_matrix"] = cam_intrinsic_mat.ravel().tolist()
    x, y, z = np.mgrid[-1:1.1:0.25, -1:1.1:0.25, -1:1.1:0.25]
    grid = np.c_[x.ravel(), y.ravel(), z.ravel()] * np.pi
    rot_matrices = Rt.from_euler("XYZ", grid).as_matrix()
    rot_mat_len = len(rot_matrices)
    org_mat = [[
		[0.9503879347691897,
		-0.12228840479714526,
		0.28602153677183806,
		0.0],
		[0.095412504779002483,
		0.98976322388070337,
		0.10613771518679699,
		0.0],
		[-0.29607301021361471,
		-0.073581972693359823,
		0.95232686926160104,
		0.0],
		[-1.8429529140825325,
		-1.2214317950963878,
		0.6428005992092698,
		1.0]
	]] * rot_mat_len
    fin_ex_mat = np.array(org_mat)
    fin_ex_mat[:, :3, :3] = rot_matrices
    for i in range(rot_mat_len):
        json_name = str(i) + ".json"
        json_cont = deepcopy(json_list)
        json_cont["extrinsic"] = fin_ex_mat[i].ravel().tolist()
        with open(json_name, 'w') as f:
            json.dump(json_cont, f)


if __name__ == "__main__":
    generate_json_file()