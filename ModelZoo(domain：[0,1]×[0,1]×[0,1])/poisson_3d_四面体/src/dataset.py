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
        xyz = xyz[xyz[:, 0] + xyz[:, 1] + xyz[:, 2] - 1 < 0]

        # ----------------边界----------------
        # 边界平面
        c1 = np.linspace(0, 1, n_b)
        c2 = np.linspace(0, 1, n_b)
        c1, c2 = np.meshgrid(c1, c2)
        c1 = c1.reshape(-1, 1)
        c2 = c2.reshape(-1, 1)
        c = np.concatenate([c1, c2], axis=1)
        c = c[c[:, 0] + c[:, 1] - 1 < 0]

        b0 = np.zeros_like(c[:, [0]])

        xyz_b1 = np.concatenate([b0.reshape(-1, 1), c[:, [0]], c[:, [1]]], axis=1)
        xyz_b2 = np.concatenate([c[:, [0]], b0.reshape(-1, 1), c[:, [1]]], axis=1)
        xyz_b3 = np.concatenate([c[:, [0]], c[:, [1]], b0.reshape(-1, 1)], axis=1)

        z4 = 1 - c[:, [0]] - c[:, [1]]
        xyz_b4 = np.concatenate([c[:, [0]], c[:, [1]], z4], axis=1)

        xyz_b = np.vstack([xyz_b1, xyz_b2, xyz_b3, xyz_b4])

        u_b = np.sin(4*np.pi * xyz_b[:, [0]]) * np.sin(4*np.pi * xyz_b[:, [1]]) * np.sin(4*np.pi * xyz_b[:, [2]]) / (
                    3 * np.pi ** 2)
        # ----------------边界----------------

        xyz = ms.Tensor(xyz, dtype=ms.float32)
        xyz_b = ms.Tensor(xyz_b, dtype=ms.float32)
        u_b = ms.Tensor(u_b, dtype=ms.float32)

        return xyz, xyz_b, u_b

