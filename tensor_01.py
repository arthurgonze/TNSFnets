# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:40:57 2021

@author: Arthur
"""
import numpy as np


x_star = np.fromfile('./sims/cube_00/timesteps/X_star.x', dtype=np.int32)
y_star = np.fromfile('./sims/cube_00/timesteps/X_star.y', dtype=np.int32)
z_star = np.fromfile('./sims/cube_00/timesteps/X_star.z', dtype=np.int32)

t_star = np.arange(0, 40, 1/60)

n = x_star.shape[0]
timesteps = t_star.shape[0] # 2400

t_star = np.reshape(t_star, (timesteps,1))


## Grid data rearrange
pos_star = np.zeros(shape=(n,3))
pos_star[:,0] = x_star
pos_star[:,1] = y_star
pos_star[:,2] = z_star


xx = np.tile(pos_star[:, 0:1], (1, timesteps))  # n x timesteps
yy = np.tile(pos_star[:, 1:2], (1, timesteps))  # n x timesteps
zz = np.tile(pos_star[:, 2:3], (1, timesteps))  # n x timesteps
tt = np.tile(t_star, (1, n)).T  # n x timesteps

x = xx.flatten()[:, None]  # NT x 1
y = yy.flatten()[:, None]  # NT x 1
z = zz.flatten()[:, None]  # NT x 1
t = tt.flatten()[:, None]  # NT x 1

## Velocities data rearrange
u_star = np.zeros(shape=(n,timesteps))
v_star = np.zeros(shape=(n,timesteps))
w_star = np.zeros(shape=(n,timesteps))
for i in range(0,2400):
    u_star[:,i] = np.fromfile('./sims/cube_00/timesteps/U_star_'+str(i)+'.u', dtype=np.float64)
    v_star[:,i] = np.fromfile('./sims/cube_00/timesteps/V_star_'+str(i)+'.v', dtype=np.float64)
    w_star[:,i] = np.fromfile('./sims/cube_00/timesteps/W_star_'+str(i)+'.w', dtype=np.float64)
    
u = u_star.flatten()# NT x 1
v = v_star.flatten()# NT x 1
w = w_star.flatten()# NT x 1

u = np.reshape(u, (u.shape[0],1))
v = np.reshape(v, (v.shape[0],1))
w = np.reshape(w, (w.shape[0],1))

## DOMINIO
domain = np.concatenate([x, y, z, t, u, v, w], 1)
domain_t0 = domain[:, :][domain[:, 3] == 0]

domain_x_lb = domain[:, :][domain[:, 0] == 0]
domain_x_up = domain[:, :][domain[:, 0] == 31]

domain_y_lb = domain[:, :][domain[:, 1] == 0]
domain_y_ub = domain[:, :][domain[:, 1] == 31]

domain_z_lb = domain[:, :][domain[:, 2] == 0]
domain_z_up = domain[:, :][domain[:, 2] == 31]

domain_boundary = np.concatenate([domain_x_lb, domain_x_up, domain_y_lb, domain_y_ub, domain_z_lb, domain_z_up], 0)

idx = np.random.choice(domain.shape[0], 980000, replace=False)


## DADOS TREINAMENTO
# random points
x_train = domain[idx, 0].reshape(domain[idx, 0].shape[0], 1)
y_train = domain[idx, 1].reshape(domain[idx, 1].shape[0], 1)
z_train = domain[idx, 2].reshape(domain[idx, 2].shape[0], 1)
t_train = domain[idx, 3].reshape(domain[idx, 3].shape[0], 1)

# initial points
x0_train = domain_t0[:, 0].reshape(domain_t0[:, 0].shape[0], 1)
y0_train = domain_t0[:, 1].reshape(domain_t0[:, 1].shape[0], 1)
z0_train = domain_t0[:, 2].reshape(domain_t0[:, 2].shape[0], 1)

t0_train = domain_t0[:, 3].reshape(domain_t0[:, 3].shape[0], 1)

u0_train = domain_t0[:, 4].reshape(domain_t0[:, 4].shape[0], 1)
v0_train = domain_t0[:, 5].reshape(domain_t0[:, 5].shape[0], 1)
w0_train = domain_t0[:, 6].reshape(domain_t0[:, 6].shape[0], 1)

# boundary points
xb_train = domain_boundary[:, 0].reshape(domain_boundary[:, 0].shape[0], 1)
yb_train = domain_boundary[:, 1].reshape(domain_boundary[:, 1].shape[0], 1)
zb_train = domain_boundary[:, 2].reshape(domain_boundary[:, 2].shape[0], 1)

tb_train = domain_boundary[:, 3].reshape(domain_boundary[:, 3].shape[0], 1)

ub_train = domain_boundary[:, 4].reshape(domain_boundary[:, 4].shape[0], 1)
vb_train = domain_boundary[:, 5].reshape(domain_boundary[:, 5].shape[0], 1)
wb_train = domain_boundary[:, 6].reshape(domain_boundary[:, 6].shape[0], 1)