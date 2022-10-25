import mindspore as ms
import numpy as np


class Trainset_poisson():
    def __init__(self, *args):
        self.args = args

    def __call__(self):
        return self.data()

    def data(self):
        n_x = self.args[0]
        n_y = self.args[1]
        n_z = self.args[2]
        n_b = self.args[3]

        # 内部点
        x = np.linspace(0, 1, n_x)
        y = np.linspace(0, 1, n_y)
        z = np.linspace(0, 1, n_z)
        x, y, z = np.meshgrid(x, y, z)
        xyz = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)))
        xyz = xyz[(xyz[:, 0] - 0.5) ** 2 + (xyz[:, 1] - 0.5) ** 2 <= 0.25]

        # ----------------边界----------------
        # 边界平面
        theta_bcs = np.linspace(0, np.pi * 2, n_b)
        r_bcs = np.array([0.5])
        z_bcs = np.linspace(0, 1, n_b)
        theta_bcs, r_bcs, z_bcs = np.meshgrid(theta_bcs, r_bcs, z_bcs)

        x_b1 = r_bcs * np.cos(theta_bcs) + 0.5
        y_b1 = r_bcs * np.sin(theta_bcs) + 0.5
        z_b1 = z_bcs

        xyz_b1 = np.concatenate([x_b1.reshape(-1, 1), y_b1.reshape(-1, 1), z_b1.reshape(-1, 1)], axis=1)

        x_b23 = np.linspace(0, 1, n_b)
        y_b23 = np.linspace(0, 1, n_b)
        x_b23, y_b23 = np.meshgrid(x_b23, y_b23)
        xy_b23 = np.hstack([x_b23.reshape(-1, 1), y_b23.reshape(-1, 1)])
        xy_b23 = xy_b23[(xy_b23[:, 0] - 0.5) ** 2 + (xy_b23[:, 1] - 0.5) ** 2 <= 0.25]

        z_b2 = np.zeros_like(xy_b23[:, [0]])
        z_b3 = np.zeros_like(xy_b23[:, [0]]) + 1

        xyz_b2 = np.hstack([xy_b23, z_b2])
        xyz_b3 = np.hstack([xy_b23, z_b3])

        xyz_b = np.vstack([xyz_b1, xyz_b2, xyz_b3])

        u_b = np.sin(4*np.pi * xyz_b[:, [0]]) * np.sin(4*np.pi * xyz_b[:, [1]]) * np.sin(4*np.pi * xyz_b[:, [2]]) / (
                    3 * np.pi ** 2)
        # ----------------边界----------------

        xyz = ms.Tensor(xyz, dtype=ms.float32)
        xyz_b = ms.Tensor(xyz_b, dtype=ms.float32)
        u_b = ms.Tensor(u_b, dtype=ms.float32)

        return xyz, xyz_b, u_b