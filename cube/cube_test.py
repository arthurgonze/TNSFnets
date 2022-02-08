# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:14:26 2021

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
# from NSFnet_default_model import VPNSFnet
# from NSFnet_saveload_model import VPNSFnet
from NSFnet_fluidborders_model import VPNSFnet


def plot_solution(X_star, data, index, points, title, xLabel, yLabel, colorbarLabel, experiment):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = points
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, data.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    
    plt.pcolor(X, Y, U_star, cmap = 'jet')
    cbar = plt.colorbar()
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('../figures/'+experiment+'/solution_'+title+".png", dpi=300, format='png')
    plt.show()
    
    
if __name__ == "__main__":
    #######################################################################
    ########################### Init Model Data ###########################
    #######################################################################
    experiment = 'cube_01'
    experiment_slice = 'y_equal_10_'
    purp = 'exact'
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
    
    
    ########################################################################
    ############################## Model Load ##############################
    ########################################################################
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
    
    
    for i in range(3, 243):
        percent = (i/243) * 100
        print("Generating predict data ... "+str(percent)+"%")
        #######################################################################
        ############################## Test Data ##############################
        #######################################################################
        purp = 'exact'
        x_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/x_star_test_'+str(i)+'.npy')
        y_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/y_star_test_'+str(i)+'.npy')
        z_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/z_star_test_'+str(i)+'.npy')
        t_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/t_star_test_'+str(i)+'.npy')
        u_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/u_star_test_'+str(i)+'.npy')
        v_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/v_star_test_'+str(i)+'.npy')
        w_star_test = np.load('./data/'+experiment+'/'+experiment_slice+purp+'/w_star_test_'+str(i)+'.npy')
        
        
        ########################################################################
        ############################## Prediction ##############################
        ########################################################################
        purp = 'pred'
        u_pred, v_pred, w_pred, p_pred = model.predict(x_star_test, y_star_test, z_star_test, t_star_test)
        
        u_pred = u_pred.reshape(u_pred.shape[0], 1)
        v_pred = v_pred.reshape(v_pred.shape[0], 1)
        w_pred = w_pred.reshape(w_pred.shape[0], 1)
        p_pred = p_pred.reshape(p_pred.shape[0], 1)
        
        np.save('./data/'+experiment+'/'+experiment_slice+purp+'/u_pred_'+str(i), u_pred)
        np.save('./data/'+experiment+'/'+experiment_slice+purp+'/v_pred_'+str(i), v_pred)
        np.save('./data/'+experiment+'/'+experiment_slice+purp+'/w_pred_'+str(i), w_pred)
        np.save('./data/'+experiment+'/'+experiment_slice+purp+'/p_pred_'+str(i), p_pred)
        
        
        
        
        # ########################################################################
        # ################################# Error ################################
        # ########################################################################
        # error_u = np.linalg.norm(u_star_test - u_pred, 2) / np.linalg.norm(u_star_test, 2)
        # error_v = np.linalg.norm(v_star_test - v_pred, 2) / np.linalg.norm(v_star_test, 2)
        # error_w = np.linalg.norm(w_star_test - w_pred, 2) / np.linalg.norm(w_star_test, 2)
        # # error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
        
        # error_u_matrix = u_star_test - u_pred
        # error_v_matrix = v_star_test - v_pred
        # error_w_matrix = w_star_test - w_pred
        # # error_p_matrix = p_star - p_pred
        
        # print('Error u: %e' % error_u)
        # print('Error v: %e' % error_v)
        # print('Error v: %e' % error_w)
        # # print('Error p: %e' % error_p)
    
    # ######################################################################
    # ############################## Plotting ##############################
    # ######################################################################  
    # v_mag = np.sqrt((u_star_test**2)+(v_star_test**2)+(w_star_test**2))
    # v_pred_mag = np.sqrt((u_pred**2)+(v_pred**2)+(w_pred**2))
    
    # pos_array = np.zeros((len(x_star_test), 2))
    # pos_array[:,0] = x_star_test[:, 0]
    # pos_array[:,1] = y_star_test[:, 0]
    # # pos_array = pos_array[pos_array[:, 0].argsort()]
    # # X, Y = np.meshgrid(pos_array[:,0], pos_array[:,1])
    
    
       
    # def plotVectors():
    #     #################### Teste x,y,u,v vectors #################### 
    #     fig, ax = plt.subplots(figsize = (10, 10))
    #     h = ax.quiver(x_star_test, y_star_test, u_star_test, v_star_test, v_mag, cmap='jet')
    #     cbar = plt.colorbar(h, ax=ax)
    #     cbar.set_label("velocity mag", rotation=270, labelpad=20)
    #     ax.set_title('Vector Field')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     ## plt.colorbar()
    #     plt.savefig('../figures/cube_00/velocities_exact_vectors.png', dpi=300, format='png')
    #     plt.show()
        
        
    #     #################### Teste x,y,u,v pred vectors ####################
    #     fig, ax = plt.subplots(figsize = (10, 10))
    #     h = ax.quiver(x_star_test, y_star_test, u_pred, v_pred, v_pred_mag, cmap='jet')
    #     cbar = plt.colorbar(h, ax=ax)
    #     cbar.set_label("velocity mag", rotation=270, labelpad=20)
    #     ax.set_title('Vector Field Pred')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     ## plt.colorbar()
    #     plt.savefig('../figures/cube_00/velocities_pred_vectors.png', dpi=300, format='png')
    #     plt.show()
        
    
    # # def plotStreamlines():
    # #     VX = griddata(pos_array, u_star.flatten(), (X, Y), method='cubic')
    # #     VY = griddata(pos_array, v_star.flatten(), (X, Y), method='cubic')
        
    # #     VX_pred = griddata(pos_array, u_pred.flatten(), (X, Y), method='cubic')
    # #     VY_pred = griddata(pos_array, v_pred.flatten(), (X, Y), method='cubic')
        
    # #     fig, ax = plt.subplots(figsize = (10, 10))
    # #     h = ax.streamplot(X, Y, VX, VY, density=1, linewidth=2, color = v_mag, cmap='jet')
    # #     ax.set_title('Vector Field - Streamlines')
    # #     cbar = plt.colorbar(h.lines, ax=ax)
    # #     cbar.set_label("velocity mag", rotation=270, labelpad=20)
    # #     plt.savefig('./figures/cube_00/velocities_exact_streamlines.png', dpi=300, format='png')
    # #     plt.show()
        
    
    
    # #     #################### Teste x,y,u,v pred streamlines #################### 
    # #     fig, ax = plt.subplots(figsize = (10, 10))
    # #     h = ax.streamplot(X, Y, VX_pred, VY_pred, density=1, linewidth=2, color = v_pred_mag, cmap='jet')
    # #     cbar = plt.colorbar(h.lines, ax=ax)
    # #     cbar.set_label("velocity mag", rotation=270, labelpad=20)
    # #     ax.set_title('Vector Field Pred- Streamlines')
    # #     plt.savefig('./figures/cube_00/velocities_pred_streamlines.png', dpi=300, format='png')
    # #     plt.show()
        
    
    # def plotColormaps():
    #     # print("not implemented")
    #     plot_solution(pos_array, u_pred, 1, 500,'U pred','x','y','u',experiment)
    #     plot_solution(pos_array, v_pred, 2, 500,'V pred','x','y','v',experiment)
    #     plot_solution(pos_array, w_pred, 3, 500,'W pred','x','y','v',experiment)
    #     plot_solution(pos_array, p_pred, 4, 500,'P pred','x','y','p',experiment) 
        
    #     plot_solution(pos_array, u_star_test, 5, 500,'U exact','x','y','u',experiment)
    #     plot_solution(pos_array, v_star_test, 6, 500,'V exact','x','y','v',experiment)
    #     plot_solution(pos_array, w_star_test, 7, 500,'W exact','x','y','v',experiment)
    #     # plot_solution(pos_array, p_star, 8, 500,'P exact','x','y','p',experiment)
        
    #     plot_solution(pos_array, error_u_matrix, 9, 500,'U exact - U predict','x','y','Ue - Up',experiment)
    #     plot_solution(pos_array, error_v_matrix, 10, 500,'V exact - V predict','x','y','Ve - Vp',experiment)
    #     plot_solution(pos_array, error_w_matrix, 11, 500,'W exact - W predict','x','y','We - Wp',experiment)
    #     # plot_solution(pos_array, error_p_matrix, 12, 500,'P exact - P predict','x','y','Pe - Pp',experiment)
    
    # # plot()
    # # plotPoints()
    # plotVectors()
    # # plotStreamlines()
    # plotColormaps()