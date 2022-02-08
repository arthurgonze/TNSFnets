# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 12:54:45 2021

@author: Arthur
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
# from NSFnet_default_model import VPNSFnet
# from NSFnet_saveload_model import VPNSFnet
from NSFnet_fluidborders_model import VPNSFnet
# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)


if __name__ == "__main__":
    experiment = 'cube_01'
    
    x_train = np.load('./data/'+experiment+'/x_train.npy')
    y_train = np.load('./data/'+experiment+'/y_train.npy')
    z_train = np.load('./data/'+experiment+'/z_train.npy')
    t_train = np.load('./data/'+experiment+'/t_train.npy')


    # initial points
    x0_train = np.load('./data/'+experiment+'/x0_train.npy')
    y0_train = np.load('./data/'+experiment+'/y0_train.npy')
    z0_train = np.load('./data/'+experiment+'/z0_train.npy')
    t0_train = np.load('./data/'+experiment+'/t0_train.npy')
    u0_train = np.load('./data/'+experiment+'/u0_train.npy')
    v0_train = np.load('./data/'+experiment+'/v0_train.npy')
    w0_train = np.load('./data/'+experiment+'/w0_train.npy')


    # boundary points
    xb_train = np.load('./data/'+experiment+'/xb_train.npy')
    yb_train = np.load('./data/'+experiment+'/yb_train.npy')
    zb_train = np.load('./data/'+experiment+'/zb_train.npy')
    tb_train = np.load('./data/'+experiment+'/tb_train.npy')
    ub_train = np.load('./data/'+experiment+'/ub_train.npy')
    vb_train = np.load('./data/'+experiment+'/vb_train.npy')
    wb_train = np.load('./data/'+experiment+'/wb_train.npy')
    
    # empty boundary points
    xbe_train = np.load('./data/'+experiment+'/xbe_train.npy')
    ybe_train = np.load('./data/'+experiment+'/ybe_train.npy')
    zbe_train = np.load('./data/'+experiment+'/zbe_train.npy')
    tbe_train = np.load('./data/'+experiment+'/tbe_train.npy')
    ube_train = np.load('./data/'+experiment+'/ube_train.npy')
    vbe_train = np.load('./data/'+experiment+'/vbe_train.npy')
    wbe_train = np.load('./data/'+experiment+'/wbe_train.npy')

    
    ######################################################################
    ############################## Training ##############################
    ######################################################################
    layers = [4, 50, 50, 50, 50, 50, 50, 50, 4]
    filename = 'cube_01_30kAdam1e-3_BFGS_7x50_2s_emptyboundaries_noDiff'
    load = True
    nndir = './models/'+filename+'.pickle'
    # pass the initial t0 training data, along with the boundary training data
    # and the random scattered points
    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     xbe_train, ybe_train, zbe_train, tbe_train,
                     ube_train, vbe_train, wbe_train,
                     x_train, y_train, z_train, t_train, layers, load, nndir)

    # The two-stage optimization (Adam with learning rate 10−3, 30.000 epochs 
    # and L-BFGS-B)
    
    # model.Adam_train(50, 1e-3)
    model.Adam_train(30000, 1e-3)
    # model.Adam_train(5000, 1e-3)
    # model.Adam_train(5000, 1e-4)
    # model.Adam_train(50000, 1e-5)
    # model.Adam_train(50000, 1e-6) 
    model.BFGS_train()
        
    model.plotLoss()
    print("Saving NSFNet")
    # self.saver.save(self.sess, './models/'+filename)
    # nndir = './models/'+filename+'.pickle'
    model.save_NN(nndir)