import open3d as o3d
import math
import numpy as np
from scipy.spatial.transform import Rotation as Rt
from visualization import VisOpen3D


def output_multi_view_images(t, pcdList, camIntrinsic, w = 1024,
                             h = 768, num = 12):
    # t is required to compute camera extrinsic!
    dist = np.linalg.norm(t)
    new_z_vec = t / dist
    new_x_vec = np.cross(new_z_vec, np.array([0, 1, 0]))
    new_y_vec = np.cross(new_z_vec, np.array([1, 0, 0]))
    new_x_vec /= np.linalg.norm(new_x_vec)
    new_y_vec /= np.linalg.norm(new_y_vec)
    init_ex_mat = np.r_[np.c_[new_x_vec, new_y_vec, new_z_vec, t],
                   [0,0,0,1]]

    # create window
    vis = VisOpen3D(width=w, height=h, visible=False)

    # point cloud
    for pcd in pcdList:
        vis.add_geometry(pcd)


    # update view

    for i in range(0, num + 1):
        # generate multi-view theta, phi
        extrinsic = np.eye(4)
        if i:
            phi = math.acos(-1.0 + (2.0 * i - 1.0) / num)
            theta = math.sqrt(num * math.pi) * phi
            print(i, theta, phi)
            rel_rot_mat = Rt.from_euler("XYZ", [0., phi, theta]).as_matrix()
            extrinsic[:3, :3] = rel_rot_mat
            extrinsic = init_ex_mat @ extrinsic


        vis.update_view_point(camIntrinsic, extrinsic)

        # save view point to file
        # vis.save_view_point("view_point.json")
        # vis.load_view_point("2.json")


        # capture images
        depth = vis.capture_depth_float_buffer(show=True)
        image = vis.capture_screen_float_buffer(show=True)

        # save to file
        vis.capture_screen_image("capture_screen_image{}.png".format(i))
        vis.capture_depth_image("capture_depth_image.png".format(i))

    '''
    # draw camera
    if window_visible:
        vis.load_view_point("view_point.json")
        intrinsic = vis.get_view_point_intrinsics()
        extrinsic = vis.get_view_point_extrinsics()
        vis.draw_camera(intrinsic, extrinsic, scale=0.5, color=[0.8, 0.2, 0.8])
        # vis.update_view_point(intrinsic, extrinsic)

    if window_visible:
        vis.load_view_point("view_point.json")
        vis.run()
    '''

    del vis


if __name__ == "__main__":
    main()
