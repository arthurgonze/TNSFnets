# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 10:45:24 2021

@author: Arthur
"""
import numpy as np
import scipy.io

data = scipy.io.loadmat('../PINNs/main/Data/cylinder_nektar_wake.mat')

u_star = data['U_star']  # n x 2 x timesteps -> (5000 x 2 x 200)
p_star = data['p_star']  # n x timesteps -> (5000 x 200)
t_star = data['t']  # timesteps x 1 -> (200 x 1)
x_star = data['X_star']  # n x 2 -> (5000 x 2)

n = x_star.shape[0] # 5000
timesteps = t_star.shape[0] # 200

# Rearrange Data
xx = np.tile(x_star[:, 0:1], (1, timesteps))  # n x timesteps
yy = np.tile(x_star[:, 1:2], (1, timesteps))  # n x timesteps
tt = np.tile(t_star, (1, n)).T  # n x timesteps

uu = u_star[:, 0, :]  # n x timesteps
vv = u_star[:, 1, :]  # n x timesteps
PP = p_star  # n x timesteps

x = xx.flatten()[:, None]  # NT x 1
y = yy.flatten()[:, None]  # NT x 1
t = tt.flatten()[:, None]  # NT x 1

u = uu.flatten()[:, None]  # NT x 1
v = vv.flatten()[:, None]  # NT x 1
p = PP.flatten()[:, None]  # NT x 1

# need add unsupervised part
# We consider a domain defined by [1, 8] × [−2, 2] and the timesteps interval is [0, 7]
data1 = np.concatenate([x, y, t, u, v, p], 1)

# timesteps interval is [0, 7]
data2 = data1[:, :][data1[:, 2] <= 7]

# [1, 8]
data3 = data2[:, :][data2[:, 0] >= 1]
data4 = data3[:, :][data3[:, 0] <= 8]

# [−2, 2]
data5 = data4[:, :][data4[:, 1] >= -2]
data_domain = data5[:, :][data5[:, 1] <= 2] # this is the final data domain

# data in timesteps t == 0
data_t0 = data_domain[:, :][data_domain[:, 2] == 0]
data_t0.shape

data_y1 = data_domain[:, :][data_domain[:, 0] == 1]
data_y8 = data_domain[:, :][data_domain[:, 0] == 8]
data_x = data_domain[:, :][data_domain[:, 1] == -2]
data_x2 = data_domain[:, :][data_domain[:, 1] == 2]

data_sup_b_train = np.concatenate([data_y1, data_y8, data_x, data_x2], 0)

idx = np.random.choice(data_domain.shape[0], 140000, replace=False)

x_train = data_domain[idx, 0].reshape(data_domain[idx, 0].shape[0], 1)
y_train = data_domain[idx, 1].reshape(data_domain[idx, 1].shape[0], 1)
t_train = data_domain[idx, 2].reshape(data_domain[idx, 2].shape[0], 1)

x0_train = data_t0[:, 0].reshape(data_t0[:, 0].shape[0], 1)
y0_train = data_t0[:, 1].reshape(data_t0[:, 1].shape[0], 1)
t0_train = data_t0[:, 2].reshape(data_t0[:, 2].shape[0], 1)
u0_train = data_t0[:, 3].reshape(data_t0[:, 3].shape[0], 1)
v0_train = data_t0[:, 4].reshape(data_t0[:, 4].shape[0], 1)

xb_train = data_sup_b_train[:, 0].reshape(data_sup_b_train[:, 0].shape[0], 1)
yb_train = data_sup_b_train[:, 1].reshape(data_sup_b_train[:, 1].shape[0], 1)
tb_train = data_sup_b_train[:, 2].reshape(data_sup_b_train[:, 2].shape[0], 1)
ub_train = data_sup_b_train[:, 3].reshape(data_sup_b_train[:, 3].shape[0], 1)
vb_train = data_sup_b_train[:, 4].reshape(data_sup_b_train[:, 4].shape[0], 1)