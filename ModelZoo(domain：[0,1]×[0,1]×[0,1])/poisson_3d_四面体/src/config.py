import argparse

class Options_poisson(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--decay_rate', type=float, default=0.8,
                            help='decay_rate in lr_scheduler for Adam optimizer')
        parser.add_argument('--step_per_epoch', type=int, default=2500,
                            help='step size in lr_scheduler for Adam optimizer')
        parser.add_argument('--decay_steps', type=int, default=1, help='衰减的step数')
        parser.add_argument('--epochs_Adam', type=int, default=100000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=1000, help='epochs for LBFGS optimizer')
        parser.add_argument('--dim_hidden', type=int, default=128, help='neurons in hidden layers')
        parser.add_argument('--hidden_layers', type=int, default=8, help='number of hidden layers')
        parser.add_argument('--n_x', type=int, default=51, help='number of interior point samples on the X-axis')
        parser.add_argument('--n_y', type=int, default=51, help='number of interior point samples on the Y-axis')
        parser.add_argument('--n_z', type=int, default=51, help='number of interior point samples on the Z-axis')
        parser.add_argument('--n_b', type=int, default=64, help='number of boundary point samples on the every axis')
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        return arg


