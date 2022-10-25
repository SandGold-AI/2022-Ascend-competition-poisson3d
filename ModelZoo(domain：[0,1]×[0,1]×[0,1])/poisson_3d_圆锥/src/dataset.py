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
        xyz = xyz[(xyz[:, 0] - 0.5) ** 2 + (xyz[:, 1] - 0.5) ** 2 <= 0.25]  # 圆柱
        r = ((xyz[:, 0] - 0.5) ** 2 + (xyz[:, 1] - 0.5) ** 2) ** (1 / 2)
        z_b = (0.5 - r) / 0.5
        xyz = xyz[xyz[:, 2] <= z_b]

        # ----------------边界----------------
        # 边界平面
        x = np.linspace(0, 1, n_b)
        y = np.linspace(0, 1, n_b)
        x, y = np.meshgrid(x, y)
        xy_b = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        xy_b = xy_b[(xy_b[:, 0] - 0.5) ** 2 + (xy_b[:, 1] - 0.5) ** 2 <= 0.25]  # 圆柱

        r = ((xy_b[:, [0]] - 0.5) ** 2 + (xy_b[:, [1]] - 0.5) ** 2) ** (1 / 2)

        z_b1 = np.zeros_like(xy_b[:, [0]])
        z_b2 = (0.5 - r) / 0.5

        xyz_b1 = np.concatenate([xy_b, z_b1], axis=1)
        xyz_b2 = np.concatenate([xy_b, z_b2], axis=1)
        xyz_b = np.vstack([xyz_b1, xyz_b2])

        u_b = np.sin(4*np.pi * xyz_b[:, [0]]) * np.sin(4*np.pi * xyz_b[:, [1]]) * np.sin(4*np.pi * xyz_b[:, [2]]) / (
                    3 * np.pi ** 2)
        # ----------------边界----------------

        xyz = ms.Tensor(xyz, dtype=ms.float32)
        xyz_b = ms.Tensor(xyz_b, dtype=ms.float32)
        u_b = ms.Tensor(u_b, dtype=ms.float32)

        return xyz, xyz_b, u_b