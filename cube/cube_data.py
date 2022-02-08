# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 13:55:04 2021

@author: Arthur
"""
import numpy as np


if __name__ == "__main__":
    #######################################################################
    ############################ Reading Data #############################
    #######################################################################
    experiment = 'cube_01'
    experiment_slice = 'y_equal_10_exact'
    x_star = np.fromfile('../sims/cube_00/timesteps/X_star.x', dtype=np.int32)
    y_star = np.fromfile('../sims/cube_00/timesteps/X_star.y', dtype=np.int32)
    z_star = np.fromfile('../sims/cube_00/timesteps/X_star.z', dtype=np.int32)
    
    t_star = np.arange(0, 40, 1/60)
    grid_size = 32*32*32
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
        u_star[:,i] = np.fromfile('../sims/'+experiment+'/timesteps/U_star_'+str(i)+'.u', dtype=np.float64)
        v_star[:,i] = np.fromfile('../sims/'+experiment+'/timesteps/V_star_'+str(i)+'.v', dtype=np.float64)
        w_star[:,i] = np.fromfile('../sims/'+experiment+'/timesteps/W_star_'+str(i)+'.w', dtype=np.float64)
        
    u = u_star.flatten()# NT x 1
    v = v_star.flatten()# NT x 1
    w = w_star.flatten()# NT x 1
    
    u = np.reshape(u, (u.shape[0],1))
    v = np.reshape(v, (v.shape[0],1))
    w = np.reshape(w, (w.shape[0],1))
    
    #######################################################################
    ############################# Data Domain #############################
    #######################################################################
    domain = np.concatenate([x, y, z, t, u, v, w], 1)
    t0 = 0.05
    t_limit = 2
    
    domain_t0 = domain[:, :][domain[:, 3] == t0]

    #######################################################################
    ############################ Training Data ############################
    #######################################################################
    
    #######################
    #### random points ####
    #######################
    amostra_aleatoria = round(grid_size*1.5)

    domain_random = domain[:, :][domain[:, 3] <= t_limit]
    domain_random = domain_random[:, :][domain_random[:, 3] > t0]
    
    # velocity must be different than 0
    domain_random = domain[:, :][domain[:, 4] > 0]
    domain_random = domain[:, :][domain[:, 5] > 0]
    domain_random = domain[:, :][domain[:, 6] > 0]
    
    domain_random = domain_random[:, :][domain_random[:, 0] > 1]
    domain_random = domain_random[:, :][domain_random[:, 0] < 30]
    
    domain_random = domain_random[:, :][domain_random[:, 1] > 1]
    domain_random = domain_random[:, :][domain_random[:, 1] < 30]
    
    domain_random = domain_random[:, :][domain_random[:, 2] > 1]
    domain_random = domain_random[:, :][domain_random[:, 2] < 30]
    
    # domain_random = domain[:, :][domain[:, 3] % 1 == 0]
    idx_random = np.random.choice(domain_random.shape[0], amostra_aleatoria, replace=False)
    x_train = domain_random[idx_random, 0].reshape(domain_random[idx_random, 0].shape[0], 1)
    y_train = domain_random[idx_random, 1].reshape(domain_random[idx_random, 1].shape[0], 1)
    z_train = domain_random[idx_random, 2].reshape(domain_random[idx_random, 2].shape[0], 1)
    t_train = domain_random[idx_random, 3].reshape(domain_random[idx_random, 3].shape[0], 1)
    
    ########################
    #### initial points ####
    ########################
    
    ## n initial points
    # idx_initial = np.random.choice(domain_t0.shape[0], 10000, replace=False)
    # x0_train = domain_t0[idx_initial, 0].reshape(domain_t0[idx_initial, 0].shape[0], 1)
    # y0_train = domain_t0[idx_initial, 1].reshape(domain_t0[idx_initial, 1].shape[0], 1)
    # z0_train = domain_t0[idx_initial, 2].reshape(domain_t0[idx_initial, 2].shape[0], 1)
    
    # t0_train = domain_t0[idx_initial, 3].reshape(domain_t0[idx_initial, 3].shape[0], 1)
    
    # u0_train = domain_t0[idx_initial, 4].reshape(domain_t0[idx_initial, 4].shape[0], 1)
    # v0_train = domain_t0[idx_initial, 5].reshape(domain_t0[idx_initial, 5].shape[0], 1)
    # w0_train = domain_t0[idx_initial, 6].reshape(domain_t0[idx_initial, 6].shape[0], 1)
    
    ## all initial points
    x0_train = domain_t0[:, 0].reshape(domain_t0[:, 0].shape[0], 1)
    y0_train = domain_t0[:, 1].reshape(domain_t0[:, 1].shape[0], 1)
    z0_train = domain_t0[:, 2].reshape(domain_t0[:, 2].shape[0], 1)
    
    t0_train = domain_t0[:, 3].reshape(domain_t0[:, 3].shape[0], 1)
    
    u0_train = domain_t0[:, 4].reshape(domain_t0[:, 4].shape[0], 1)
    v0_train = domain_t0[:, 5].reshape(domain_t0[:, 5].shape[0], 1)
    w0_train = domain_t0[:, 6].reshape(domain_t0[:, 6].shape[0], 1)
    
    
    #########################
    #### boundary points ####
    #########################
    
    # todos os tempos pares
    # domain_boundary_limited = domain_boundary[:, :][domain_boundary[:, 3] % 3 == 0]
    
    ## n boundary points
    domain_boundary = domain[:, :][domain[:, 3] <= t_limit]
    domain_x_lb = domain_boundary[:, :][domain_boundary[:, 0] == 1]
    domain_x_up = domain_boundary[:, :][domain_boundary[:, 0] == 30]
    
    domain_y_lb = domain_boundary[:, :][domain_boundary[:, 1] == 1]
    domain_y_ub = domain_boundary[:, :][domain_boundary[:, 1] == 30]
    
    domain_z_lb = domain_boundary[:, :][domain_boundary[:, 2] == 1]
    domain_z_up = domain_boundary[:, :][domain_boundary[:, 2] == 30]
    
    domain_boundary_limited = np.concatenate([domain_x_lb, domain_x_up, domain_y_lb, domain_y_ub, domain_z_lb, domain_z_up], 0)
    
    amostra_bordas = round(grid_size*1.5)
    
    idx_boundary = np.random.choice(domain_boundary_limited.shape[0], amostra_bordas, replace=False)
    xb_train = domain_boundary_limited[idx_boundary, 0].reshape(domain_boundary_limited[idx_boundary, 0].shape[0], 1)
    yb_train = domain_boundary_limited[idx_boundary, 1].reshape(domain_boundary_limited[idx_boundary, 1].shape[0], 1)
    zb_train = domain_boundary_limited[idx_boundary, 2].reshape(domain_boundary_limited[idx_boundary, 2].shape[0], 1)
    
    tb_train = domain_boundary_limited[idx_boundary, 3].reshape(domain_boundary_limited[idx_boundary, 3].shape[0], 1)
    
    ub_train = domain_boundary_limited[idx_boundary, 4].reshape(domain_boundary_limited[idx_boundary, 4].shape[0], 1)
    vb_train = domain_boundary_limited[idx_boundary, 5].reshape(domain_boundary_limited[idx_boundary, 5].shape[0], 1)
    wb_train = domain_boundary_limited[idx_boundary, 6].reshape(domain_boundary_limited[idx_boundary, 6].shape[0], 1)
    
    ## all boundary points
    # xb_train = domain_boundary_limited[:, 0].reshape(domain_boundary_limited[:, 0].shape[0], 1)
    # yb_train = domain_boundary_limited[:, 1].reshape(domain_boundary_limited[:, 1].shape[0], 1)
    # zb_train = domain_boundary_limited[:, 2].reshape(domain_boundary_limited[:, 2].shape[0], 1)
    
    # tb_train = domain_boundary_limited[:, 3].reshape(domain_boundary_limited[:, 3].shape[0], 1)
    
    # ub_train = domain_boundary_limited[:, 4].reshape(domain_boundary_limited[:, 4].shape[0], 1)
    # vb_train = domain_boundary_limited[:, 5].reshape(domain_boundary_limited[:, 5].shape[0], 1)
    # wb_train = domain_boundary_limited[:, 6].reshape(domain_boundary_limited[:, 6].shape[0], 1)
    
    ############################
    #### empty fluid points ####
    ############################
    amostra_vazia = round(grid_size*1.5)

    domain_empty = domain[:, :][domain[:, 3] <= t_limit]
    domain_empty = domain_empty[:, :][domain_empty[:, 3] > t0]
    
    # velocity must be equal to 0
    domain_empty = domain_empty[:, :][domain_empty[:, 4] == 0]
    domain_empty = domain_empty[:, :][domain_empty[:, 5] == 0]
    domain_empty = domain_empty[:, :][domain_empty[:, 6] == 0]
    
    domain_empty = domain_empty[:, :][domain_empty[:, 0] > 1]
    domain_empty = domain_empty[:, :][domain_empty[:, 0] < 30]
    
    domain_empty = domain_empty[:, :][domain_empty[:, 1] > 1]
    domain_empty = domain_empty[:, :][domain_empty[:, 1] < 30]
    
    domain_empty = domain_empty[:, :][domain_empty[:, 2] > 1]
    domain_empty = domain_empty[:, :][domain_empty[:, 2] < 30]
    
    # domain_random = domain[:, :][domain[:, 3] % 1 == 0]
    idx_empty = np.random.choice(domain_empty.shape[0], amostra_vazia, replace=False)
    xbe_train = domain_empty[idx_empty, 0].reshape(domain_empty[idx_empty, 0].shape[0], 1)
    ybe_train = domain_empty[idx_empty, 1].reshape(domain_empty[idx_empty, 1].shape[0], 1)
    zbe_train = domain_empty[idx_empty, 2].reshape(domain_empty[idx_empty, 2].shape[0], 1)
    tbe_train = domain_empty[idx_empty, 3].reshape(domain_empty[idx_empty, 3].shape[0], 1)
    ube_train = domain_empty[idx_empty, 4].reshape(domain_empty[idx_empty, 4].shape[0], 1)
    vbe_train = domain_empty[idx_empty, 5].reshape(domain_empty[idx_empty, 5].shape[0], 1)
    wbe_train = domain_empty[idx_empty, 6].reshape(domain_empty[idx_empty, 6].shape[0], 1)
    
    
    #######################################################################
    ############################## Test Data ##############################
    #######################################################################
    # for i in range(3, 243): #range(t0, 2*t_limit)
    #     percent = (i/243) * 100
    #     print("Generating test data ... "+str(percent)+"%")
        
    #     domain_t = domain[:, :][domain[:, 3] == t_star[i]]
    #     # domain_t = domain_t[:, :][domain_t[:, 2] == 15] #z==15
    #     # domain_t = domain_t[:, :][domain_t[:, 1] == 15] #y==15
    #     domain_t = domain_t[:, :][domain_t[:, 1] == 10] #y==10
    #     x_star_test = domain_t[:, 0].reshape(domain_t[:, 0].shape[0], 1)
    #     y_star_test = domain_t[:, 1].reshape(domain_t[:, 1].shape[0], 1)
    #     z_star_test = domain_t[:, 2].reshape(domain_t[:, 2].shape[0], 1)
        
    #     t_star_test = domain_t[:, 3].reshape(domain_t[:, 3].shape[0], 1)
        
    #     u_star_test = domain_t[:, 4].reshape(domain_t[:, 4].shape[0], 1)
    #     v_star_test = domain_t[:, 5].reshape(domain_t[:, 5].shape[0], 1)
    #     w_star_test = domain_t[:, 6].reshape(domain_t[:, 6].shape[0], 1)
        
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/x_star_test_'+str(i), x_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/y_star_test_'+str(i), y_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/z_star_test_'+str(i), z_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/t_star_test_'+str(i), t_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/u_star_test_'+str(i), u_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/v_star_test_'+str(i), v_star_test)
    #     np.save('./data/'+experiment+'/'+experiment_slice+'/w_star_test_'+str(i), w_star_test)
    
    
    #######################################################################
    ############################# Saving Data #############################
    #######################################################################
    ## training data
    # random points
    np.save('./data/'+experiment+'/x_train', x_train)
    np.save('./data/'+experiment+'/y_train', y_train)
    np.save('./data/'+experiment+'/z_train', z_train)
    np.save('./data/'+experiment+'/t_train', t_train)
    
    
    # initial points
    np.save('./data/'+experiment+'/x0_train', x0_train)
    np.save('./data/'+experiment+'/y0_train', y0_train)
    np.save('./data/'+experiment+'/z0_train', z0_train)
    np.save('./data/'+experiment+'/t0_train', t0_train)
    np.save('./data/'+experiment+'/u0_train', u0_train)
    np.save('./data/'+experiment+'/v0_train', v0_train)
    np.save('./data/'+experiment+'/w0_train', w0_train)
    
    
    # boundary points
    np.save('./data/'+experiment+'/xb_train', xb_train)
    np.save('./data/'+experiment+'/yb_train', yb_train)
    np.save('./data/'+experiment+'/zb_train', zb_train)
    np.save('./data/'+experiment+'/tb_train', tb_train)
    np.save('./data/'+experiment+'/ub_train', ub_train)
    np.save('./data/'+experiment+'/vb_train', vb_train)
    np.save('./data/'+experiment+'/wb_train', wb_train)
    
    # empty points
    np.save('./data/'+experiment+'/xbe_train', xbe_train)
    np.save('./data/'+experiment+'/ybe_train', ybe_train)
    np.save('./data/'+experiment+'/zbe_train', zbe_train)
    np.save('./data/'+experiment+'/tbe_train', tbe_train)
    np.save('./data/'+experiment+'/ube_train', ube_train)
    np.save('./data/'+experiment+'/vbe_train', vbe_train)
    np.save('./data/'+experiment+'/wbe_train', wbe_train)