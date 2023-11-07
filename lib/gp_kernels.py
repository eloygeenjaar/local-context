# ported from: https://github.com/googleinterns/local_global_ts_representation/blob/main/gl_rep/gp_kernel.py
""""Kernel functions used for the Gaussian Process prior"""

import math
import torch


def periodic_kernel(T, length_scale, period, sigma_var=1.):
    xs = torch.arange(T).float()
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = (xs_in-xs_out).abs()
    periodic_dist = torch.sin(distance_matrix*math.pi/period)
    distance_matrix_scaled = 2 * (periodic_dist**2) / (length_scale ** 2)
    kernel_matrix = (-distance_matrix_scaled).exp()
    return sigma_var*kernel_matrix

def rbf_kernel(T, length_scale, sigma_var=1.):
    xs = torch.arange(T).float()
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = torch.norm(xs_in - xs_out).pow(2)
    distance_matrix_scaled = distance_matrix / (2*(length_scale ** 2))
    kernel_matrix = (-distance_matrix_scaled).exp()
    return sigma_var*kernel_matrix

def matern_kernel(T, length_scale):
    xs = torch.arange(T).float()
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = (xs_in - xs_out).abs()
    distance_matrix_scaled = distance_matrix / length_scale.sqrt()
    kernel_matrix = (-distance_matrix_scaled).exp()
    return kernel_matrix

def cauchy_kernel(T, sigma, length_scale):
    xs = torch.arange(T).floaT()
    xs_in = xs.unsqueeze(0)
    xs_out = xs.unsqueeze(1)
    distance_matrix = torch.norm(xs_in - xs_out.float).pow(2)
    distance_matrix_scaled = distance_matrix / (length_scale ** 2)
    kernel_matrix = torch.div(sigma, (distance_matrix_scaled + 1.))

    alpha = 0.001
    eye = torch.eye(n=kernel_matrix.size(0))
    return kernel_matrix + alpha * eye
