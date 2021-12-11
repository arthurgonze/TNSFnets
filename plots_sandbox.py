# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:27:48 2021

@author: Arthur
"""
import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
# from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.figure_factory as ff

Re = 40 # Reinalds number
lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

x = np.linspace(-0.5, 1.0, 101)
y = np.linspace(-0.5, 1.5, 101)
np.random.seed(1234)

x_star = (np.random.rand(1000, 1) - 1 / 3) * 3 / 2
y_star = (np.random.rand(1000, 1) - 1 / 4) * 2
u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)

pos_array = np.zeros((len(x_star), 2))
pos_array[:,0] = x_star[:, 0] 
pos_array[:,1] = y_star[:, 0]
pos_array = pos_array[pos_array[:, 1].argsort()]

# X, Y = np.meshgrid(x, y, sparse=False, indexing='xy')
# lin_x = np.linspace(x_star.min(), x_star.max(), 101)
# lin_y = np.linspace(y_star.min(), y_star.max(), 101)
# newX, newY = np.meshgrid(lin_x, lin_y, sparse=False, indexing='xy')
# lin_u = 1 - np.exp(lam * newX) * np.cos(2 * np.pi * newY)
# lin_v = (lam / (2 * np.pi)) * np.exp(lam * newX) * np.sin(2 * np.pi * newY)


# v_mag = np.sqrt((u_star**2)+(v_star**2))
# ################## Teste x,y,u,v vectors ################## 
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.quiver(x_star, y_star, u_star, v_star, v_mag, cmap='jet', scale=30)
# ax.set_title('Vector Field')
# plt.axis('equal')
# plt.show()



# v_mag = np.sqrt((lin_u**2)+(lin_v**2))
# ################## Teste x,y,u,v streamlines################## 
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.streamplot(newX, newY, lin_u, lin_v, density=1, linewidth=2, color=v_mag, cmap='jet')
# ax.set_title('Vector Field - Streamlines')
# plt.axis('equal')
# plt.show()


# lossIterations = np.linspace(0, 1, 10)
# for it in range(10):
#     lossIterations[it] = np.cos(it)
# loss_x = np.linspace(0, 1, 10)

# fig, ax = plt.subplots(figsize = (10, 10))
# ax.plot(loss_x, lossIterations);
# ax.set_title('Loss')
# # plt.axis('equal')
# plt.show()
