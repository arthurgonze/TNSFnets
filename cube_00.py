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
# from plotting import newfig, savefig
import time
# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)

#############################################
###################VP NSFnet#################
#############################################

class VPNSFnet:
    # Initialize the class
    def __init__(self, x0, y0, z0, t0, u0, v0, w0, xb, yb, zb, tb, ub, vb, wb, x, y, z, t, layers, load, filename):
        X0temp = np.concatenate([x0, y0, z0, t0], 1)  # remove the second bracket
        Xbtemp = np.concatenate([xb, yb, zb, tb], 1)
        Xtemp = np.concatenate([x, y, z, t], 1)

        self.lowb = Xbtemp.min(0)  # minimal number in each column
        self.upb = Xbtemp.max(0)

        self.X0 = X0temp
        self.Xb = Xbtemp
        self.X = Xtemp

        self.x0 = X0temp[:, 0:1]
        self.y0 = X0temp[:, 1:2]
        self.z0 = X0temp[:, 2:3]
        self.t0 = X0temp[:, 3:4]

        self.xb = Xbtemp[:, 0:1]
        self.yb = Xbtemp[:, 1:2]
        self.zb = Xbtemp[:, 2:3]
        self.tb = Xbtemp[:, 3:4]

        self.x = Xtemp[:, 0:1]
        self.y = Xtemp[:, 1:2]
        self.z = Xtemp[:, 2:3]
        self.t = Xtemp[:, 3:4]

        self.u0 = u0
        self.v0 = v0
        self.w0 = w0

        self.ub = ub
        self.vb = vb
        self.wb = wb

        self.layers = layers
        
        # Initialize NN
        # tf placeholders and graph
        
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
            
        
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        
        self.x_ini_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
        self.y_ini_tf = tf.placeholder(tf.float32, shape=[None, self.y0.shape[1]])
        self.z_ini_tf = tf.placeholder(tf.float32, shape=[None, self.z0.shape[1]])
        self.t_ini_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])
        self.u_ini_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])
        self.v_ini_tf = tf.placeholder(tf.float32, shape=[None, self.v0.shape[1]])
        self.w_ini_tf = tf.placeholder(tf.float32, shape=[None, self.w0.shape[1]])

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.z_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.zb.shape[1]])
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.tb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])
        self.w_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.wb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.z_tf = tf.placeholder(tf.float32, shape=[None, self.z.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])

        
        self.lossIterations = []
        self.leIterations = []
        self.liIterations = []
        self.lbIterations = []

        self.weights, self.biases = self.initialize_NN(layers)
        
        # pred initial cond
        self.u_ini_pred, self.v_ini_pred, self.w_ini_pred, self.p_ini_pred = \
            self.net_NS(self.x_ini_tf, self.y_ini_tf, self.z_ini_tf, self.t_ini_tf)
        
        # pred boundary cond
        self.u_boundary_pred, self.v_boundary_pred, self.w_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf, self.z_boundary_tf, self.t_boundary_tf)
        
        # pred random
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf)

        # The weighting coefficients of the loss function are both fixed in 
        # this case: α=β=100
        self.alpha = 100
        self.beta = 100

        # set loss function
        # initial cond loss
        self.li = tf.reduce_mean(tf.square(self.u_ini_tf - self.u_ini_pred)) + \
            tf.reduce_mean(tf.square(self.v_ini_tf - self.v_ini_pred)) + \
                tf.reduce_mean(tf.square(self.w_ini_tf - self.w_ini_pred))
                
        # boundary cond loss
        self.lb = tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
            tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                tf.reduce_mean(tf.square(self.w_boundary_tf - self.w_boundary_pred))
                
        # PDE loss        
        self.le = tf.reduce_mean(tf.square(self.f_u_pred)) + \
            tf.reduce_mean(tf.square(self.f_v_pred)) + \
                tf.reduce_mean(tf.square(self.f_w_pred)) + \
                    tf.reduce_mean(tf.square(self.f_e_pred))
        
        # loss func
        self.loss = self.alpha * self.li + \
                    self.beta * self.lb + \
                    self.le

        # set optimizer
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        # self.saver
        
        if load:
            print("Loading NSFNet")
            self.loader = tf.train.import_meta_graph('./models/'+filename+'.meta')
            self.loader.restore(self.sess, tf.train.latest_checkpoint('./models/'))
            
        init = tf.global_variables_initializer()
        self.sess.run(init)
        if not load:
            self.saver = tf.train.Saver()
            print("Saving NSFNet")
            self.saver.save(self.sess, './models/'+filename)

    # do not need adaptation
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # do not need adaptation
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # ###################without assume###############
    # supervised train
    def net_NS(self, x, y, z, t):

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        return u, v, w, p

    # unsupervised train
    def net_f_NS(self, x, y, z, t):

        Re = 1

        u_v_w_p = self.neural_net(tf.concat([x, y, z, t], 1), self.weights, self.biases)
        u = u_v_w_p[:, 0:1]
        v = u_v_w_p[:, 1:2]
        w = u_v_w_p[:, 2:3]
        p = u_v_w_p[:, 3:4]

        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]

        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]

        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1/Re * (u_xx + u_yy + u_zz)
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1/Re * (v_xx + v_yy + v_zz)
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)
        f_e = u_x + v_y + w_z

        return u, v, w, p, f_u, f_v, f_w, f_e


    # Precisa remover lambda_1
    def callback(self, loss):
        print('Loss: %.3e' % loss)

    # O tf_dict do treinamento precisa ser modificado
    def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0, self.z_ini_tf: self.z0, self.t_ini_tf: self.t0,
                   self.u_ini_tf: self.u0, self.v_ini_tf: self.v0, self.w_ini_tf: self.w0,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t, self.learning_rate: learning_rate}

        # tf_dict deve ser o processo de alimentação de dados
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()
                self.lossIterations.append(loss_value)
                self.lbIterations.append(self.sess.run(self.lb, tf_dict))
                self.liIterations.append(self.sess.run(self.li, tf_dict))
                self.leIterations.append(self.sess.run(self.le, tf_dict))
                
            # checkpoints
            if it % 100 == 0:
                self.saver.save(self.sess, './models/'+filename)
                
        # final save
        self.saver.save(self.sess, './models/'+filename)
                

    def BFGS_train(self):

        tf_dict = {self.x_ini_tf: self.x0, self.y_ini_tf: self.y0, self.z_ini_tf: self.z0, self.t_ini_tf: self.t0,
                   self.u_ini_tf: self.u0, self.v_ini_tf: self.v0, self.w_ini_tf: self.w0,
                   self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb, self.z_boundary_tf: self.zb,
                   self.t_boundary_tf: self.tb, self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.w_boundary_tf: self.wb, self.x_tf: self.x, self.y_tf: self.y, self.z_tf: self.z,
                   self.t_tf: self.t}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
        # save every 100 iterations
        self.saver.save(self.sess, './models/'+filename, global_step=100)


    # Não há necessidade de alterar, você pode precisar prestar atenção a x_tf, etc.
    def predict(self, x_star, y_star, z_star, t_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        w_star = self.sess.run(self.w_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, w_star, p_star

    def plotLoss(self):
        
        loss_x = np.linspace(0, 1, len(self.lossIterations))
        
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(loss_x, self.lossIterations)
        ax.set_title('Loss')
        plt.savefig('./figures/cube_00/loss.png', dpi=300, format='png')
        plt.show()
        
        plt.plot(loss_x, self.leIterations)
        plt.title('Le loss')
        plt.savefig('./figures/cube_00/le_loss.png', dpi=300, format='png')
        plt.show()
        
        
        plt.plot(loss_x, self.liIterations, '--g', label = 'li')
        plt.plot(loss_x, self.lbIterations, '-b', label = 'lb')
        plt.title('Li and Lb losses')
        plt.legend()
        plt.savefig('./figures/cube_00/li_lb_losses.png', dpi=300, format='png')
        plt.show()


def plot_solution(X_star, data, index, points, title, xLabel, yLabel, colorbarLabel):
    
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
    plt.savefig('./figures/cube_00/solution_'+title+".png", dpi=300, format='png')
    plt.show()
    


if __name__ == "__main__":
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

    
    ######################################################################
    ############################## Training ##############################
    ######################################################################
    layers = [4, 50, 50, 50, 50, 50, 50, 50, 4]
    filename = 'cube_00_adam_50_1e-3'
    load = False
    # pass the initial t0 training data, along with the boundary training data
    # and the random scattered points
    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, layers, load, filename)

    # The two-stage optimization (Adam with learning rate 10−3, 30.000 epochs 
    # and L-BFGS-B)
    if not load:
        model.Adam_train(50, 1e-3)
        # model.Adam_train(30000, 1e-3)
        # model.BFGS_train()
    
    #######################################################################
    ############################## Test Data ##############################
    #######################################################################
    domain_t39 = domain[:, :][domain[:, 3] == 39.9]
    x_star_test = domain_t39[:, 0].reshape(domain_t39[:, 0].shape[0], 1)
    y_star_test = domain_t39[:, 1].reshape(domain_t39[:, 1].shape[0], 1)
    z_star_test = domain_t39[:, 2].reshape(domain_t39[:, 2].shape[0], 1)

    t_star_test = domain_t39[:, 3].reshape(domain_t39[:, 3].shape[0], 1)

    u_star_test = domain_t39[:, 4].reshape(domain_t39[:, 4].shape[0], 1)
    v_star_test = domain_t39[:, 5].reshape(domain_t39[:, 5].shape[0], 1)
    w_star_test = domain_t39[:, 6].reshape(domain_t39[:, 6].shape[0], 1)
    
    ########################################################################
    ############################## Prediction ##############################
    ########################################################################
    u_pred, v_pred, w_pred, p_pred = model.predict(x_star_test, y_star_test, z_star_test, t_star_test)
    
    u_pred = u_pred.reshape(u_pred.shape[0], 1)
    v_pred = v_pred.reshape(v_pred.shape[0], 1)
    w_pred = w_pred.reshape(w_pred.shape[0], 1)
    p_pred = p_pred.reshape(p_pred.shape[0], 1)
    

    ########################################################################
    ################################# Error ################################
    ########################################################################
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_w = np.linalg.norm(w_star - w_pred, 2) / np.linalg.norm(w_star, 2)
    # error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    
    error_u_matrix = u_star - u_pred
    error_v_matrix = v_star - v_pred
    error_w_matrix = w_star - w_pred
    # error_p_matrix = p_star - p_pred

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error v: %e' % error_w)
    # print('Error p: %e' % error_p)
    
    ######################################################################
    ############################## Plotting ##############################
    ######################################################################  
    v_mag = np.sqrt((u_star**2)+(v_star**2)+(w_star**2))
    v_pred_mag = np.sqrt((u_pred**2)+(v_pred**2)+(w_pred**2))
    
    pos_array = np.zeros((len(x_star_test), 2))
    pos_array[:,0] = x_star_test[:, 0]
    pos_array[:,1] = y_star_test[:, 0]
    pos_array = pos_array[pos_array[:, 0].argsort()]
    X, Y = np.meshgrid(pos_array[:,0], pos_array[:,1])
    

       
    def plotVectors():
        #################### Teste x,y,u,v vectors #################### 
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.quiver(x_star_test, y_star_test, u_star_test, v_star_test, v_mag, cmap='jet')
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field')
        ## plt.colorbar()
        plt.savefig('./figures/cube_00/velocities_exact_vectors.png', dpi=300, format='png')
        plt.show()
        
        
        #################### Teste x,y,u,v pred vectors ####################
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.quiver(x_star_test, y_star_test, u_pred, v_pred, v_pred_mag, cmap='jet')
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field Pred')
        ## plt.colorbar()
        plt.savefig('./figures/cube_00/velocities_pred_vectors.png', dpi=300, format='png')
        plt.show()
        
    
    def plotStreamlines():
        VX = griddata(pos_array, u_star.flatten(), (X, Y), method='cubic')
        VY = griddata(pos_array, v_star.flatten(), (X, Y), method='cubic')
        
        VX_pred = griddata(pos_array, u_pred.flatten(), (X, Y), method='cubic')
        VY_pred = griddata(pos_array, v_pred.flatten(), (X, Y), method='cubic')
        
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.streamplot(X, Y, VX, VY, density=1, linewidth=2, color = v_mag, cmap='jet')
        ax.set_title('Vector Field - Streamlines')
        cbar = plt.colorbar(h.lines, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        plt.savefig('./figures/cube_00/velocities_exact_streamlines.png', dpi=300, format='png')
        plt.show()
        
    
    
        #################### Teste x,y,u,v pred streamlines #################### 
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.streamplot(X, Y, VX_pred, VY_pred, density=1, linewidth=2, color = v_pred_mag, cmap='jet')
        cbar = plt.colorbar(h.lines, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field Pred- Streamlines')
        plt.savefig('./figures/cube_00/velocities_pred_streamlines.png', dpi=300, format='png')
        plt.show()
        
    
    def plotColormaps():
        # print("not implemented")
        plot_solution(pos_array, u_pred, 1, 500,'U pred','x','y','u')
        plot_solution(pos_array, v_pred, 2, 500,'V pred','x','y','v')
        plot_solution(pos_array, w_pred, 3, 500,'W pred','x','y','v')
        plot_solution(pos_array, p_pred, 4, 500,'P pred','x','y','p') 
        
        plot_solution(pos_array, u_star, 5, 500,'U exact','x','y','u')
        plot_solution(pos_array, v_star, 6, 500,'V exact','x','y','v')
        plot_solution(pos_array, w_star, 7, 500,'W exact','x','y','v')
        # plot_solution(pos_array, p_star, 8, 500,'P exact','x','y','p')
        
        plot_solution(pos_array, error_u_matrix, 9, 500,'U exact - U predict','x','y','Ue - Up')
        plot_solution(pos_array, error_v_matrix, 10, 500,'V exact - V predict','x','y','Ve - Vp')
        plot_solution(pos_array, error_w_matrix, 11, 500,'W exact - W predict','x','y','We - Wp')
        # plot_solution(pos_array, error_p_matrix, 12, 500,'P exact - P predict','x','y','Pe - Pp')
    
    # plot()
    # plotPoints()
    plotVectors()
    # plotStreamlines()
    plotColormaps()
    # model.plotLoss()