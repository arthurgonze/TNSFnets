# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 22:14:16 2021

@author: Arthur
"""

import sys
sys.path.append('../PINNs/Utilities')

import tensorflow as tf
import numpy as np
import time
import matplotlib.pyplot as plt
# from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable



# set random seed
np.random.seed(1234)
tf.set_random_seed(1234)

class VPNSFnet:
    # Initialize the class
    def __init__(self, xb, yb, ub, vb, x, y, layers):
        # remove the second bracket
        Xb = np.concatenate([xb, yb], 1)
        X = np.concatenate([x, y], 1)

        self.lowb = Xb.min(0)   # min number in each column
        self.upb = Xb.max(0)    # max number in each column

        self.Xb = Xb
        self.X = X

        self.xb = Xb[:, 0:1]
        self.yb = Xb[:, 1:2]
        self.x = X[:, 0:1]
        self.y = X[:, 1:2]

        self.ub = ub
        self.vb = vb

        self.layers = layers
        
        self.lossIterations = np.linspace(0, 1, 100)
        self.leIterations = np.linspace(0, 1, 100)
        self.lbIterations = np.linspace(0, 1, 100)

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(layers)

        self.learning_rate = tf.placeholder(tf.float32, shape=[])

        # Initialize parameters #
        # self.alpha = tf.Variable([0.0], dtype=tf.float32) # dynamic weighting

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                               log_device_placement=True))

        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.xb.shape[1]])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.yb.shape[1]])
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.ub.shape[1]])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, self.vb.shape[1]])

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, self.y.shape[1]])

        self.u_boundary_pred, self.v_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf)
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf)

        # The fixed weighting coefficient α=100 for boundary conditions is chosen for training these NSFnets.
        self.alpha = 100
        
        # need adaptation, set loss function
        # L = Le + αLb + βLi
        self.Lb = ((tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)))+(tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred))))
        self.Le = tf.reduce_mean(tf.square(self.f_u_pred)) + \
            tf.reduce_mean(tf.square(self.f_v_pred)) + \
            tf.reduce_mean(tf.square(self.f_e_pred))
        
        self.loss = (self.alpha * self.Lb) + self.Le

        # do not need adaptation #
        self.optimizer_Adam = tf.train.AdamOptimizer(self.learning_rate)# add learning rate here
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

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

    # do not need adaptation
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    # do not need adaptation
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lowb) / (self.upb - self.lowb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))#activation function tanh?
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    # need adaptation, supervised data-driven
    def net_NS(self, x, y):
        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]

        return u, v, p

    # need adaptation, unsupervised train
    def net_f_NS(self, x, y):
        u_v_p = self.neural_net(tf.concat([x, y], 1), self.weights, self.biases)
        u = u_v_p[:, 0:1]
        v = u_v_p[:, 1:2]
        p = u_v_p[:, 2:3]
        
        print("##############################################################")
        print("x: " + str(x))
        print("u_v_p: "+ str(u_v_p))
        print("u: " + str(u))
        print("##############################################################")
        

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = (u * u_x + v * u_y) + p_x - (1.0/40) * (u_xx + u_yy)# Navier-Stokes equation for u(x velocity), residual governing equation 1
        f_v = (u * v_x + v * v_y) + p_y - (1.0/40) * (v_xx + v_yy)# Navier-Stokes equation for v(y velocity), residual governing equation 2
        f_e = u_x + v_y# incompressibility constraint

        return u, v, p, f_u, f_v, f_e
    
    # do not need adaptation
    def callback(self, loss):
        print('Loss: %.3e' % loss)

    # do not need adaptation
    def Adam_train(self, nIter=5000, learning_rate=1e-3):

        tf_dict = {self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y, self.learning_rate: learning_rate}

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
                self.lossIterations[int(it/10)] = loss_value
                self.lbIterations[int(it/10)] = self.sess.run(self.Lb, tf_dict)
                self.leIterations[int(it/10)] = self.sess.run(self.Le, tf_dict)
                

    # do not need adaptation
    def BFGS_train(self):

        tf_dict = {self.x_boundary_tf: self.xb, self.y_boundary_tf: self.yb,
                   self.u_boundary_tf: self.ub, self.v_boundary_tf: self.vb,
                   self.x_tf: self.x, self.y_tf: self.y}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
        
    # do not need adaptation
    def predict(self, x_star, y_star):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star}

        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)

        return u_star, v_star, p_star
    
    def plotLoss(self):
        loss_x = np.linspace(0, 1, 100)
        
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(loss_x, self.lossIterations)
        ax.set_title('Loss')
        plt.savefig('./figures/kf_2D/loss.png', dpi=300, format='png')
        plt.show()
        
        plt.plot(loss_x, self.leIterations, ':r', label = 'le')
        plt.plot(loss_x, self.lbIterations, '-b', label = 'lb')
        plt.title('Le and Lb losses')
        plt.legend()
        plt.savefig('./figures/kf_2D/lb_le_losses.png', dpi=300, format='png')
        plt.show()
    
    
def plot_solution(X_star, data, index, points, title, xLabel, yLabel, colorbarLabel):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = points
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[0], ub[0], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, data.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    cbar = plt.colorbar()
    cbar.set_label(colorbarLabel, rotation=270, labelpad=20)
    plt.title(title)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.savefig('./figures/kf_2D/solution_'+title+".png", dpi=300, format='png')
    plt.show()
    

if __name__ == "__main__":
    # There is no initial condition for this steady flow
    ## supervised ##
    # For computing the equation loss of NSFnets, 2,601 points are randomly selected inside the domain
    N_train = 2601
    # best neural network architecture for VP-NSFnet with 2,601 residual points is 7×100.
    layers = [2, 100, 100, 100, 100, 100, 100, 100, 3]

    # Load Data
    Re = 40 # Reinalds number
    lam = 0.5 * Re - np.sqrt(0.25 * (Re ** 2) + 4 * (np.pi ** 2))

    # We consider a computational domain of [−0.5, 1.0] × [−0.5, 1.5].
    # There are 101 points with fixed spatial coordinate on each boundary, Nb=4×101
    x = np.linspace(-0.5, 1.0, 101)
    y = np.linspace(-0.5, 1.5, 101)

    # Nb Points
    yb1 = np.array([-0.5] * 100)
    yb2 = np.array([1] * 100)
    xb1 = np.array([-0.5] * 100)
    xb2 = np.array([1.5] * 100)

    y_train1 = np.concatenate([y[1:101], y[0:100], xb1, xb2], 0)
    x_train1 = np.concatenate([yb1, yb2, x[0:100], x[1:101]], 0)
    
    xb_train = x_train1.reshape(x_train1.shape[0], 1)
    yb_train = y_train1.reshape(y_train1.shape[0], 1)
    ub_train = 1 - np.exp(lam * xb_train) * np.cos(2 * np.pi * yb_train) # u solution
    vb_train = lam / (2 * np.pi) * np.exp(lam * xb_train) * np.sin(2 * np.pi * yb_train) # v solution

    x_train = (np.random.rand(N_train, 1) - 1 / 3) * 3 / 2
    y_train = (np.random.rand(N_train, 1) - 1 / 4) * 2

    
    
    # NN Creation
    model = VPNSFnet(xb_train, yb_train, ub_train, vb_train,
                     x_train, y_train, layers)

    # train model
    # In this section, we use 3×10^4 Adam iterations with learning rate 10^−3 before the L-BFGS-B training
    model.Adam_train(750, 1e-3)
    # model.Adam_train(10000, 1e-3)
    # model.Adam_train(10000, 1e-3)
    # model.BFGS_train()

    # Test Data
    np.random.seed(1234)

    x_star = (np.random.rand(1000, 1) - 1 / 3) * 3 / 2
    y_star = (np.random.rand(1000, 1) - 1 / 4) * 2
    
    pos_array = np.zeros((len(x_star), 2))
    pos_array[:,0] = x_star[:, 0] 
    pos_array[:,1] = y_star[:, 0]
    pos_array = pos_array[pos_array[:, 1].argsort()]
    x_star[:, 0] = pos_array[:,0]
    y_star[:, 0] = pos_array[:,1]

    u_star = 1 - np.exp(lam * x_star) * np.cos(2 * np.pi * y_star)
    v_star = (lam / (2 * np.pi)) * np.exp(lam * x_star) * np.sin(2 * np.pi * y_star)
    p_star = 0.5 * (1 - np.exp(2 * lam * x_star))

    # Prediction
    
    u_pred, v_pred, p_pred = model.predict(x_star, y_star)

    # Error
    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    error_v = np.linalg.norm(v_star - v_pred, 2) / np.linalg.norm(v_star, 2)
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error p: %e' % error_p)
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    ################## Training x,y ################## 
    plt.plot(x_train, y_train, 'ob', label = 'Random', markersize=1)
    plt.plot(xb_train, yb_train, 'xr', label = 'Boundary', markersize=1)
    plt.title('Training Points')
    plt.legend(loc = 'upper right')
    plt.savefig('./figures/kf_2D/training_points.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Teste x,y ################## 
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(x_star, y_star, 'o', color='black');
    ax.set_title('Test Points')
    plt.savefig('./figures/kf_2D/test_points.png', dpi=300, format='png')
    plt.show()
    
    
    v_mag = np.sqrt((u_star**2)+(v_star**2))
    ################## Relacao pontos de treino e teste ##################
    quantidade = [len(x_train), len(xb_train), len(x_star)]
    pontos = ['T. Random', 'T. Borda', 'Teste']
    plt.bar(pontos, quantidade)
    plt.title('Training Points')
    plt.savefig('./figures/kf_2D/training_test_points_bars.png', dpi=300, format='png')
    plt.show()
    
    
    
    ################## Teste x,y,u,v vectors ################## 
    fig, ax = plt.subplots(figsize = (10, 10))
    h = ax.quiver(x_star, y_star, u_star, v_star, v_mag, cmap='jet', scale=30)
    cbar = plt.colorbar(h.lines, ax=ax)
    cbar.set_label("velocity mag", rotation=270, labelpad=20)
    ax.set_title('Vector Field')
    ## plt.colorbar()
    plt.savefig('./figures/kf_2D/velocities_exact_vectors.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Teste x,y,u,v pred vectors ##################
    v_pred_mag = np.sqrt((u_pred**2)+(v_mag**2))
    fig, ax = plt.subplots(figsize = (10, 10))
    h = ax.quiver(x_star, y_star, u_pred, v_pred, v_pred_mag, cmap='jet', scale=30)
    cbar = plt.colorbar(h.lines, ax=ax)
    cbar.set_label("velocity mag", rotation=270, labelpad=20)
    ax.set_title('Vector Field Pred')
    ## plt.colorbar()
    plt.savefig('./figures/kf_2D/velocities_pred_vectors.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Teste x,y,u,v streamlines################## 
    lin_x = np.linspace(x_star.min(), x_star.max(), 101)
    lin_y = np.linspace(y_star.min(), y_star.max(), 101)
    newX, newY = np.meshgrid(lin_x, lin_y, sparse=False, indexing='xy')
    
    lin_u = 1 - np.exp(lam * newX) * np.cos(2 * np.pi * newY)
    lin_v = (lam / (2 * np.pi)) * np.exp(lam * newX) * np.sin(2 * np.pi * newY)
    
    v_mag = np.sqrt((lin_u**2)+(lin_v**2))
    
    fig, ax = plt.subplots(figsize = (10, 10))
    h = ax.streamplot(newX, newY, lin_u, lin_v, density=1, linewidth=2, color = v_mag, cmap='jet')
    ax.set_title('Vector Field - Streamlines')
    # plt.axis('equal')
    cbar = plt.colorbar(h.lines, ax=ax)
    cbar.set_label("velocity mag", rotation=270, labelpad=20)
    plt.savefig('./figures/kf_2D/velocities_exact_streamlines.png', dpi=300, format='png')
    plt.show()
    
    
    
    ################## Training points in exact stream line plot################## 
    plt.plot(x_train, y_train, 'ob', label = 'Random', markersize=1)
    plt.plot(xb_train, yb_train, 'xr', label = 'Boundary', markersize=1)
    plt.streamplot(newX, newY, lin_u, lin_v, density=0.75, linewidth=1, color='black')
    plt.title('Training Points - Field Streamlines')
    plt.legend(loc = 'upper right')
    plt.savefig('./figures/kf_2D/points_train_exact_streamlines.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Test points in exact stream line plot ################## 
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(x_star, y_star, 'o', color='blue');
    plt.streamplot(newX, newY, lin_u, lin_v, density=0.75, linewidth=1, color='black')
    ax.set_title('Test Points - Field Streamlines')
    plt.savefig('./figures/kf_2D/points_test_exact_streamlines.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Teste x,y,u,v pred streamlines################## 
    nn = 1000
    lin_x = np.linspace(x_star.min(), x_star.max(), nn)
    lin_y = np.linspace(y_star.min(), y_star.max(), nn)
    
    X, Y = np.meshgrid(lin_x,lin_y)
    VX = griddata(pos_array, u_pred.flatten(), (X, Y), method='cubic')
    VY = griddata(pos_array, v_pred.flatten(), (X, Y), method='cubic')
    # VX, VY = np.meshgrid(u_pred.T,v_pred)
    
    teste_predX = VX
    teste_predY = VY
    
    v_pred_mag = np.sqrt((VX**2)+(VY**2))
    
    fig, ax = plt.subplots(figsize = (10, 10))
    h = ax.streamplot(X, Y, VX, VY, density=1, linewidth=2, color = v_pred_mag, cmap='jet')
    cbar = plt.colorbar(h.lines, ax=ax)
    cbar.set_label("velocity mag", rotation=270, labelpad=20)
    ax.set_title('Vector Field Pred- Streamlines')
    plt.savefig('./figures/kf_2D/velocities_pred_streamlines.png', dpi=300, format='png')
    plt.show()
    
    
    ################## Test points in predict stream line plot ################## 
    fig, ax = plt.subplots(figsize = (10, 10))
    ax.plot(x_star, y_star, 'o', color='blue');
    plt.streamplot(X, Y, VX, VY, density=0.75, linewidth=1, color='black')
    ax.set_title('Test Points - Pred Field Streamlines')
    plt.savefig('./figures/kf_2D/points_test_pred_streamlines.png', dpi=300, format='png')
    plt.show()
    
    
    
    ################## Errors ################## 
    model.plotLoss()
    
    plot_solution(pos_array, u_pred, 1, 500,'U pred','x','y','u')
    plot_solution(pos_array, v_pred, 2, 500,'V pred','x','y','v')
    plot_solution(pos_array, p_pred, 3, 500,'P pred','x','y','p') 
    
    plot_solution(pos_array, u_star, 1, 500,'U exact','x','y','u')
    plot_solution(pos_array, v_star, 2, 500,'V exact','x','y','v')
    plot_solution(pos_array, p_star, 4, 500,'P exact','x','y','p')
    
    
    plot_solution(pos_array, u_star - u_pred, 5, 500,'U exact - U predict','x','y','Ue - Up')
    plot_solution(pos_array, v_star - v_pred, 5, 500,'V exact - V predict','x','y','Ve - Vp')
    plot_solution(pos_array, p_star - p_pred, 5, 500,'P exact - P predict','x','y','Pe - Pp')
    
    
    

    
