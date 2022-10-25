import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

class Modified_MLP(nn.Cell):
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers, init_name='XavierUniform'):
        super(Modified_MLP, self).__init__()
        self.hidden_layers = hidden_layers

        self.activation = nn.Tanh()

        self.model = nn.CellList([nn.Dense(dim_in, dim_hidden, weight_init=init_name), self.activation])
        for i in range(hidden_layers):
            self.model.append(nn.Dense(dim_hidden, dim_hidden, weight_init=init_name))
            self.model.append(self.activation)
        self.model.append(nn.Dense(dim_hidden, dim_out, weight_init=init_name))

        self.fc_U = nn.Dense(dim_in, dim_hidden, weight_init=init_name)
        self.fc_V = nn.Dense(dim_in, dim_hidden, weight_init=init_name)
        self.encoder_U = nn.SequentialCell([self.fc_U, self.activation])
        self.encoder_V = nn.SequentialCell([self.fc_V, self.activation])

        self.mul = ops.Mul()

    def construct(self, x, y, z):
        x = ms.numpy.concatenate([x, y, z], axis=1)
        U = self.encoder_U(x)
        V = self.encoder_V(x)
        for i in range(self.hidden_layers):
            x = self.model[2 * i](x)  # 调用线性层
            x = self.model[2 * i + 1](x)  # 调用激活层
            x = self.mul((1 - x), U) + self.mul(x, V)  # 特征融合
        x = self.model[-1](x)  # 调用最后一个线性层得到输出

        return x


class MLP(nn.Cell):
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers, init_name='XavierUniform'):
        super(MLP, self).__init__()
        self.hidden_layers = hidden_layers

        self.activation = nn.Tanh()

        self.model = nn.CellList([nn.Dense(dim_in, dim_hidden, weight_init=init_name), self.activation])
        for i in range(hidden_layers):
            self.model.append(nn.Dense(dim_hidden, dim_hidden, weight_init=init_name))
            self.model.append(self.activation)
        self.model.append(nn.Dense(dim_hidden, dim_out, weight_init=init_name))

        self.mul = ops.Mul()

    def construct(self, x, y):
        x = ms.numpy.concatenate([x, y], axis=1)

        for i in range(self.hidden_layers):
            x = self.model[2 * i](x)  # 调用线性层
            x = self.model[2 * i + 1](x)  # 调用激活层
        x = self.model[-1](x)  # 调用最后一个线性层得到输出

        return x