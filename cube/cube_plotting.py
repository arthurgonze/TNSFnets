# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 12:30:33 2021

@author: Arthur
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import math



def plotVectors(axis1, axis2, v1, v2, v_mag, i, addr, purp):
    #################### Teste x,y,u,v vectors #################### 
    fig, ax = plt.subplots(figsize = (10, 10))
    h = ax.quiver(axis1, axis2, v1, v2, v_mag, cmap='jet')
    cbar = plt.colorbar(h, ax=ax)
    cbar.set_label("velocity mag", rotation=270, labelpad=20)
    ax.set_title('Vector Field '+ purp + ' - U e W - ' + str(i))
    plt.xlabel('x')
    plt.ylabel('z')
    ## plt.colorbar()
    plt.savefig(addr+'velocities_'+purp+'_vectors_'+str(i)+'.png', dpi=300, format='png')
    
    if i%10==0:
        plt.show()
    
if __name__ == "__main__":
    experiment = 'cube_01'
    experiment_slice = 'y_equal_10_'
    purp = 'exact'
    
    
    # exact experiment plot
    # for i in range(3, 243):
    #     percent = (i/243) * 100
    #     print("Generating exact simulation plots ... "+str(percent)+"%")
    #     purp = 'exact'
    #     x_star_test = np.load('./data/'+'cube_01'+'/'+experiment_slice+purp+'/x_star_test_'+str(i)+'.npy')
    #     # y_star_test = np.load('./data/'+'cube_01'+'/'+experiment_slice+purp+'/y_star_test_'+str(i)+'.npy')
    #     z_star_test = np.load('./data/'+'cube_01'+'/'+experiment_slice+purp+'/z_star_test_'+str(i)+'.npy')
    #     # t_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/t_star_test_'+str(i)+'.npy')
    #     u_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/u_star_test_'+str(i)+'.npy')
    #     # v_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/v_star_test_'+str(i)+'.npy')
    #     w_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/w_star_test_'+str(i)+'.npy')
        
    #     # velocity mag
    #     v_mag = np.sqrt((u_star_test**2)+(w_star_test**2))
        
    #     # Normalize the arrows:
    #     # mag2d = np.sqrt(u_star_test**2 + w_star_test**2)
    #     U = u_star_test / v_mag;
    #     # V = v_star_test / v_mag;
    #     W = w_star_test / v_mag;
        
    #     # vector plot
    #     addr = '../figures/'+experiment+'/'+experiment_slice+purp+'/'
    #     plotVectors(x_star_test, z_star_test, U, W, v_mag, i, addr, purp)
    
    # nn prediction plot
    for i in range(3, 243):
        percent = (i/243) * 100
        print("Generating predict simulation plots ... "+str(percent)+"%")
        purp = 'exact'
        x_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/x_star_test_'+str(i)+'.npy')
        # y_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/y_star_test_'+str(i)+'.npy')
        z_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/z_star_test_'+str(i)+'.npy')
        
        purp = 'pred'
        u_pred = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/u_pred_'+str(i)+'.npy')
        # v_pred = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/v_pred_'+str(i)+'.npy')
        w_pred = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/w_pred_'+str(i)+'.npy')
        
        # velocity mag
        v_mag = np.sqrt((u_pred**2)+(w_pred**2))
        
        # Normalize the arrows:
        mag2d = np.sqrt(u_pred**2 + w_pred**2)
        Upred = u_pred / mag2d;
        # Vpred = v_pred / mag2d;
        Wpred = w_pred / mag2d;
        
        # vector plot
        # addr = '../figures/'+experiment+'/'+experiment_slice+purp+'/'
        addr = '../figures/'+experiment+'/'+experiment_slice+purp+'_noDiff'+'/'
        
        plotVectors(x_star_test, z_star_test, Upred, Wpred, v_mag, i, addr, purp)