# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 21:29:31 2021

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
    def __init__(self, x0, y0, z0, t0, u0, v0, w0, xb, yb, zb, tb, ub, vb, wb, x, y, z, t, tensorfield, layers, load, filename):
        X0temp = np.concatenate([x0, y0, z0, t0], 1)  # remove the second bracket
        Xbtemp = np.concatenate([xb, yb, zb, tb], 1)
        Xtemp = np.concatenate([x, y, z, t], 1)

        self.lowb = Xbtemp.min(0)  # minimal number in each column
        # print("##############################################################")
        # print("self.lowb: " + str(self.lowb))
        # print("Xbtemp.min(0): " + str(Xbtemp.min(0)))
        # print("##############################################################")
        self.upb = Xbtemp.max(0)

        self.X0 = X0temp
        self.Xb = Xbtemp
        self.X = Xtemp

        self.x0 = X0temp[:, 0:1]
        # print("##############################################################")
        # print("self.x0: " + str(self.x0))
        # print("X0temp[:, 0:1]: " + str(X0temp[:, 0:1]))
        # print("##############################################################")
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

        
        # self.tensorfield = tf.convert_to_tensor(tensorfield)
        self.tensorfield = tensorfield
        # print(tensorfield[1,1,1,:,:])
        # print(self.tensorfield[1,1,1,:,:])
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
        
        self.tensorfield_tf = tf.placeholder(tf.float32, shape=self.tensorfield.shape)
        # print("##############################################################")
        # print("self.x.shape[1]: " + str(self.x.shape[1]))
        # print("self.x.shape: " + str(self.x.shape))
        # print("self.x: " + str(self.x))
        # print("self.x_tf.shape: " + str(self.x_tf.shape))
        # print("self.x_tf: " + str(self.x_tf))
        # print("##############################################################")
        
        self.lossIterations = []
        self.leIterations = []
        self.liIterations = []
        self.lbIterations = []

        self.weights, self.biases = self.initialize_NN(layers)
        
        # pred initial cond
        self.u_ini_pred, self.v_ini_pred, self.w_ini_pred, self.p_ini_pred = \
            self.net_NS(self.x_ini_tf, self.y_ini_tf, self.z_ini_tf, self.t_ini_tf, self.tensorfield_tf)
        
        # pred boundary cond
        self.u_boundary_pred, self.v_boundary_pred, self.w_boundary_pred, self.p_boundary_pred = \
            self.net_NS(self.x_boundary_tf, self.y_boundary_tf, self.z_boundary_tf, self.t_boundary_tf, self.tensorfield_tf)
        
        # pred random
        self.u_pred, self.v_pred, self.w_pred, self.p_pred, self.f_u_pred, self.f_v_pred, self.f_w_pred, self.f_e_pred = \
            self.net_f_NS(self.x_tf, self.y_tf, self.z_tf, self.t_tf, self.tensorfield_tf)

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
    def net_NS(self, x, y, z, t, tensorfield):

        print("##############################################################")
        print("x: " + str(x) + ", x.shape: " + str(x.shape))
        print("##############################################################")
        
        entrada = tf.concat([x, y, z, t, tensorfield], 1)
        print("##############################################################")
        print("entrada: "+str(entrada))
        print("##############################################################")
        u_v_w_p_field = self.neural_net(entrada, self.weights, self.biases)
        u = u_v_w_p_field[:, 0:1]
        v = u_v_w_p_field[:, 1:2]
        w = u_v_w_p_field[:, 2:3]
        p = u_v_w_p_field[:, 3:4]
        field = u_v_w_p_field[:, 4:5]

        return u, v, w, p

    # unsupervised train - PDE
    def net_f_NS(self, x, y, z, t, tensorfield):
        Re = 1
        rho = 1
        Beta = 1
        gravidade = -10
        entrada = tf.concat([x, y, z, t, tensorfield], 1)
        u_v_w_p_field = self.neural_net(entrada, self.weights, self.biases)
        u = u_v_w_p_field[:, 0:1]
        v = u_v_w_p_field[:, 1:2]
        w = u_v_w_p_field[:, 2:3]
        p = u_v_w_p_field[:, 3:4]
        field = u_v_w_p_field[:, 4:5]
        
        tensor = field[x, y, z, :, :]
        
        # print("##############################################################")
        # print("x: " +str(x)+", x.shape: "+str(x.shape))
        # print("tensorfield: "+str(tensorfield)+", tensorfield.shape: "+str(tensorfield.shape))
        # print("field: "+str(field)+", field.shape: "+str(field.shape))
        # print("Tensor: "+str(tensor)+", field.shape: "+str(tensor.shape))
        # print("##############################################################")
        
        a = tensor[0,0]
        b = tensor[0,1]
        c = tensor[0,2]
        d = tensor[1,0]
        e = tensor[1,1]
        f = tensor[1,2]
        g = tensor[2,0]
        h = tensor[2,1]
        i = tensor[2,2]
        
        ######################################################################
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_z = tf.gradients(u, z)[0]
        
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]
        u_zz = tf.gradients(u_z, z)[0]
        
        ut_x = a*u_x+b*u_y+c*u_z
        ut_y = d*u_x+e*u_y+f*u_z
        ut_z = g*u_x+h*u_y+i*u_z
        
        ut_xx = tf.gradients(ut_x, x)[0]
        ut_yy = tf.gradients(ut_y, y)[0]
        ut_zz = tf.gradients(ut_z, z)[0]

        ######################################################################
        v_t = tf.gradients(v, t)[0]
        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_z = tf.gradients(v, z)[0]
        
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]
        v_zz = tf.gradients(v_z, z)[0]
        
        vt_x = a*v_x+b*v_y+c*v_z
        vt_y = d*v_x+e*v_y+f*v_z
        vt_z = g*v_x+h*v_y+i*v_z
        
        vt_xx = tf.gradients(vt_x, x)[0]
        vt_yy = tf.gradients(vt_y, y)[0]
        vt_zz = tf.gradients(vt_z, z)[0]

        ######################################################################
        w_t = tf.gradients(w, t)[0]
        w_x = tf.gradients(w, x)[0]
        w_y = tf.gradients(w, y)[0]
        w_z = tf.gradients(w, z)[0]
        
        w_xx = tf.gradients(w_x, x)[0]
        w_yy = tf.gradients(w_y, y)[0]
        w_zz = tf.gradients(w_z, z)[0]
        
        wt_x = a*w_x+b*w_y+c*w_z
        wt_y = d*w_x+e*w_y+f*w_z
        wt_z = g*w_x+h*w_y+i*w_z
        
        wt_xx = tf.gradients(wt_x, x)[0]
        wt_yy = tf.gradients(wt_y, y)[0]
        wt_zz = tf.gradients(wt_z, z)[0]

        ######################################################################
        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]
        p_z = tf.gradients(p, z)[0]

        
        ## du/dt + (u dot grad)*u + (1/d)grad_t p + (Beta*u - T*u)- MODIFICADA sem difusao
        ## d = densidade, p = pressao, u = velocidade, T = tensor, Beta = fator de escala
        f_u = u_t + (u * u_x + v * u_y + w * u_z) + (1/rho)*(a*p_x + b*p_y + c*p_z) + (Beta*u - (a*u + b*v + c*w))
        f_v = v_t + (u * v_x + v * v_y + w * v_z) + (1/rho)*(d*p_x + e*p_y + f*p_z) + (Beta*v - (d*u + e*v + f*w))
        f_w = w_t + (u * w_x + v * w_y + w * w_z) + (1/rho)*(g*p_x + h*p_y + i*p_z) + (Beta*w - (g*u + h*v + i*w)) #- gravidade
        
        ## du/dt + (u dot grad)*u + (1/d)grad_t p + (Beta*u - T*u) - (1/Re)(grad^2 u)- MODIFICADA com difusao
        # f_u = u_t + (u * u_x + v * u_y + w * u_z) + (1/rho)*(a*p_x + b*p_y + c*p_z) + (Beta*u - (a*u + b*v + c*w)) - 1/Re * (ut_xx + ut_yy + ut_zz)
        # f_v = v_t + (u * v_x + v * v_y + w * v_z) + (1/rho)*(d*p_x + e*p_y + f*p_z) + (Beta*v - (d*u + e*v + f*w)) - 1/Re * (vt_xx + vt_yy + vt_zz)
        # f_w = w_t + (u * w_x + v * w_y + w * w_z) + (1/rho)*(g*p_x + h*p_y + i*p_z) + (Beta*w - (g*u + h*v + i*w)) - 1/Re * (wt_xx + wt_yy + wt_zz) - gravidade
        
        ## du/dt + (u dot grad)*u + grad p - (1/Re)(grad^2 u) - ORIGINAL
        # f_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - 1/Re * (u_xx + u_yy + u_zz)
        # f_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - 1/Re * (v_xx + v_yy + v_zz)
        # f_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - 1/Re * (w_xx + w_yy + w_zz)
        
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
                   self.t_tf: self.t, self.learning_rate: learning_rate, self.tensorfield:self.tensorfield}

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
                   self.t_tf: self.t, self.tensorfield:self.tensorfield}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
        # save every 100 iterations
        self.saver.save(self.sess, './models/'+filename, global_step=100)


    # Não há necessidade de alterar, você pode precisar prestar atenção a x_tf, etc.
    def predict(self, x_star, y_star, z_star, t_star, tensorfield):

        tf_dict = {self.x_tf: x_star, self.y_tf: y_star, self.z_tf: z_star, self.t_tf: t_star, self.tensorfield: tensorfield}

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
        plt.savefig('./figures/beltrami_3D/loss.png', dpi=300, format='png')
        plt.show()
        
        plt.plot(loss_x, self.leIterations)
        plt.title('Le loss')
        plt.savefig('./figures/beltrami_3D/le_loss.png', dpi=300, format='png')
        plt.show()
        
        
        plt.plot(loss_x, self.liIterations, '--g', label = 'li')
        plt.plot(loss_x, self.lbIterations, '-b', label = 'lb')
        plt.title('Li and Lb losses')
        plt.legend()
        plt.savefig('./figures/beltrami_3D/li_lb_losses.png', dpi=300, format='png')
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
    plt.savefig('./figures/beltrami_3D/solution_'+title+".png", dpi=300, format='png')
    plt.show()
    


if __name__ == "__main__":
    # Load Data
    def data_generate(x, y, z, t):
        a, d = 1, 1
        u = - a * (np.exp(a * x) * np.sin(a * y + d * z) + np.exp(a * z) * np.cos(a * x + d * y)) * np.exp(- d * d * t)
        v = - a * (np.exp(a * y) * np.sin(a * z + d * x) + np.exp(a * x) * np.cos(a * y + d * z)) * np.exp(- d * d * t)
        w = - a * (np.exp(a * z) * np.sin(a * x + d * y) + np.exp(a * y) * np.cos(a * z + d * x)) * np.exp(- d * d * t)
        p = - 0.5 * a * a * (np.exp(2 * a * x) + np.exp(2 * a * y) + np.exp(2 * a * z) +
                             2 * np.sin(a * x + d * y) * np.cos(a * z + d * x) * np.exp(a * (y + z)) +
                             2 * np.sin(a * y + d * z) * np.cos(a * x + d * y) * np.exp(a * (z + x)) +
                             2 * np.sin(a * z + d * x) * np.cos(a * y + d * z) * np.exp(a * (x + y))) * np.exp(-2 * d * d * t)
        return u, v, w, p
    
    def generate_tensorfield(x,y,z):
        tensorfield = np.ndarray(shape=(x, y, z, 3, 3), dtype=float)
        # tensorfield = tnp.zeros(shape=(x, y, z, 3, 3), dtype=tnp.float32)
        
        for i in range(x):
            for j in range(y):
                for k in range(z):
                    # a1x   a2x   a3x
                    # a1y   a2y   a3y
                    # a1z   a2z   a3z
                    # P = np.ndarray(shape=(3, 3), dtype=float) # autovetores
                    P = np.zeros(shape=(3, 3), dtype=np.float32)
                    # P = tnp.zeros(shape=(3, 3), dtype=tnp.float32)
                    P[0, 0] = 1
                    P[1, 1] = 1
                    P[2, 2] = 1
                    
                    # lambda1   0       0
                    # 0         lambda2 0
                    # 0         0       lambda3
                    # D = np.ndarray(shape=(3, 3), dtype=float) # autovalores
                    D = np.zeros(shape=(3, 3), dtype=np.float32)
                    # D = tnp.zeros(shape=(3, 3), dtype=tnp.float32)
                    D[0, 0] = 1
                    D[1, 1] = 1
                    D[2, 2] = 1
                    
                    tensorfield[i,j,k] = P*D*P.T
                    # k = P.T*D*P
        return tensorfield
    

    ######################################################################
    ######################### Network Structure ##########################
    ######################################################################
    # we considered two NN sizes 4 × 50 and 7 × 50.
    # layers = [4, 50, 50, 50, 50, 4]
    # entrada: x,y,z,t,tensorfield
    # saida: u,v,w,p
    layers = [5, 50, 50, 50, 50, 50, 50, 50, 4]
    
    
    ######################################################################
    ####################### Data Sampling Strategy #######################
    ######################################################################
    # batch of 10,000 points in the spatio-temporal domain is used for the equations.
    N_train = 10000
    
    # the computational domain is defined by [−1, 1] × [−1, 1] × [−1, 1]. For 
    # the training data, 31×31 points on each face are used for boundary conditions
    x1 = np.linspace(0, 60, 31,dtype=int) # map 31 points linearly from -1 to 1 in x1
    y1 = np.linspace(0, 60, 31,dtype=int) # map 31 points linearly from -1 to 1 in y1
    z1 = np.linspace(0, 60, 31,dtype=int) # map 31 points linearly from -1 to 1 in z1
    
    t1 = np.linspace(0, 1, 11)  # and the time interval is [0, 1]
    
    b0 = np.array([0] * 900)   # lower boundary
    b1 = np.array([60] * 900)    # upper boundary

    xt = np.tile(x1[0:30], 30)  # repeat x1 30 times in xt
    yt = np.tile(y1[0:30], 30)  # repeat y1 30 times in yt
    zt = np.tile(z1[0:30], 30)  # repeat z1 30 times in zt
    
    xt1 = np.tile(x1[1:31], 30) # repeat x1 one step ahead 30 times in xt1
    yt1 = np.tile(y1[1:31], 30) # repeat y1 one step ahead 30 times in yt1
    zt1 = np.tile(z1[1:31], 30) # repeat z1 one step ahead 30 times in zt1

    xr = x1[0:30].repeat(30)    # repeat every number of x1 30 times in sequence to xr
    yr = y1[0:30].repeat(30)    # repeat every number of y1 30 times in sequence to yr
    zr = z1[0:30].repeat(30)    # repeat every number of z1 30 times in sequence to zr
    
    xr1 = x1[1:31].repeat(30)   # repeat every number of x1 one step ahead 30 times in sequence to xr
    yr1 = y1[1:31].repeat(30)   # repeat every number of y1 one step ahead 30 times in sequence to yr
    zr1 = z1[1:31].repeat(30)   # repeat every number of z1 one step ahead 30 times in sequence to zr

    # concatenate all this array reapeating each one 11 times before concatenation
    # train1x = np.concatenate([b1, b0, xt1, xt, xt1, xt], 0).repeat(t1.shape[0]) 
    # train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0]) # original
    # train1z = np.concatenate([zr, zr1, zr, zr1, b1, b0], 0).repeat(t1.shape[0])
    
    ######################################################################
    ####################### Boundary Training Data #######################
    ######################################################################
    train1x = np.concatenate([b1, b0, xt1, xt, xr1, xr], 0).repeat(t1.shape[0]) 
    train1y = np.concatenate([yt, yt1, b1, b0, yr1, yr], 0).repeat(t1.shape[0])
    train1z = np.concatenate([zt1, zt, zr1, zr, b1, b0], 0).repeat(t1.shape[0])
    
    train1t = np.tile(t1, 5400) # Repeat t1 5400 times into trainlt

    # generate boundary training data
    train1ub, train1vb, train1wb, train1pb = data_generate(train1x, train1y, train1z, train1t)

    
    # xb_train = train1x.reshape(train1x.shape[0], 1)
    # yb_train = train1x.reshape(train1y.shape[0], 1)
    # zb_train = train1x.reshape(train1z.shape[0], 1)
    # tb_train = train1x.reshape(train1t.shape[0], 1)
    # ub_train = train1x.reshape(train1ub.shape[0], 1)
    # vb_train = train1x.reshape(train1vb.shape[0], 1)
    # wb_train = train1x.reshape(train1wb.shape[0], 1)
    # pb_train = train1x.reshape(train1pb.shape[0], 1)
    
    # reshape boundary training data to trainable arrays
    xb_train = train1x.reshape(train1x.shape[0], 1)
    yb_train = train1y.reshape(train1y.shape[0], 1)
    zb_train = train1z.reshape(train1z.shape[0], 1)
    tb_train = train1t.reshape(train1t.shape[0], 1)
    ub_train = train1ub.reshape(train1ub.shape[0], 1)
    vb_train = train1vb.reshape(train1vb.shape[0], 1)
    wb_train = train1wb.reshape(train1wb.shape[0], 1)
    pb_train = train1pb.reshape(train1pb.shape[0], 1)

    ######################################################################
    ####################### Initial Sampling Data ########################
    ######################################################################
    # training grid is like a truth table
    x_0 = np.tile(x1, 31 * 31) # reapeat x1 31x31 times
    y_0 = np.tile(y1.repeat(31), 31)
    z_0 = z1.repeat(31 * 31) # linear -1 to 1, reapeating every number of z1 31x31 times
    t_0 = np.array([0] * x_0.shape[0]) # snapshot at t0 = 0

    # generate initial t0 data
    u_0, v_0, w_0, p_0 = data_generate(x_0, y_0, z_0, t_0)

    # reshape initial t0 trainable data to usable arrays
    u0_train = u_0.reshape(u_0.shape[0], 1)
    v0_train = v_0.reshape(v_0.shape[0], 1)
    w0_train = w_0.reshape(w_0.shape[0], 1)
    p0_train = p_0.reshape(p_0.shape[0], 1)
    x0_train = x_0.reshape(x_0.shape[0], 1)
    y0_train = y_0.reshape(y_0.shape[0], 1)
    z0_train = z_0.reshape(z_0.shape[0], 1)
    t0_train = t_0.reshape(t_0.shape[0], 1)
    
    ######################################################################
    ################ Random Spatio-Temporal Sampling Data ################
    ######################################################################
    # Random scattered points throughout the PDE
    xx = np.random.randint(31, size=N_train) # interval of -1 to 1
    yy = np.random.randint(31, size=N_train) # interval of -1 to 1
    zz = np.random.randint(31, size=N_train) # interval of -1 to 1
    tt = np.random.randint(11, size=N_train) / 10     # interval of 0 to 1

    # generate random scattered data
    uu, vv, ww, pp = data_generate(xx, yy, zz, tt)

    # generate an workable array with that data
    x_train = xx.reshape(xx.shape[0], 1)
    y_train = yy.reshape(yy.shape[0], 1)
    z_train = zz.reshape(zz.shape[0], 1)
    t_train = tt.reshape(tt.shape[0], 1)

    
    ######################################################################
    ############################## Training ##############################
    ######################################################################
    filename = 'adam_50_1e-3'
    load = False
    tensorfield = generate_tensorfield(31, 31, 31)
    # print(tensorfield[1,1,1,:,:])
    # pass the initial t0 training data, along with the boundary training data
    # and the random scattered points
    model = VPNSFnet(x0_train, y0_train, z0_train, t0_train,
                     u0_train, v0_train, w0_train,
                     xb_train, yb_train, zb_train, tb_train,
                     ub_train, vb_train, wb_train,
                     x_train, y_train, z_train, t_train, tensorfield, layers, load, filename)

    # The two-stage optimization (Adam with learning rate 10−3, 30.000 epochs 
    # and L-BFGS-B)
    if not load:
        model.Adam_train(50, 1e-3)
        # model.Adam_train(30000, 1e-3)
        # model.BFGS_train()
    
    #######################################################################
    ############################## Test Data ##############################
    #######################################################################
    # Test Data, 1000 random points
    # x_star = (np.random.rand(1000, 1) - 1 / 2) * 2 # interval of -1 to 1
    # y_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    # z_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    # t_star = np.random.randint(11, size=(1000, 1)) / 10 # interval of 0 to 1
    
    # # generate test data
    # u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)
    
    # Test Data at t=1.00 on the plane z=0.
    # x_star = np.linspace(-1, 1, 31)
    # y_star = np.linspace(-1, 1, 31)
    # z_star = np.array([0] * 31)
    # t_star = np.array([1] * 1000)
    
    # ok but too heavy
    # x_star = np.tile(np.linspace(-1, 1, 31), 31 * 31) # reapeat x1 31x31 times
    # y_star = np.tile(np.linspace(-1, 1, 31).repeat(31), 31)
    # z_star = np.array([0] * 31).repeat(31 * 31) # linear -1 to 1, reapeating every number of z1 31x31 times
    # t_star = np.array([1] * x_star.shape[0]) # snapshot at t0 = 0
    
    x_star = np.tile(np.linspace(-1, 1, 31), 31) # reapeat x1 31x31 times
    y_star = np.tile(np.linspace(-1, 1, 31).repeat(31),1)
    z_star = np.array([0] * 31).repeat(31) # linear -1 to 1, reapeating every number of z1 31x31 times
    t_star = np.array([1] * x_star.shape[0]) # snapshot at t0 = 0
    
    # x_star = (np.random.rand(1000, 1) - 1 / 2) * 2 # interval of -1 to 1
    # y_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    # z_star = (np.random.rand(1000, 1) - 1 / 2) * 2
    # t_star = np.random.randint(11, size=(1000, 1)) / 10 # interval of 0 to 1
    
    # generate test data
    u_star, v_star, w_star, p_star = data_generate(x_star, y_star, z_star, t_star)
    
    u_star = u_star.reshape(u_star.shape[0], 1)
    v_star = v_star.reshape(v_star.shape[0], 1)
    w_star = w_star.reshape(w_star.shape[0], 1)
    p_star = p_star.reshape(p_star.shape[0], 1)
    
    ########################################################################
    ############################## Prediction ##############################
    ########################################################################
    # x_star = x_star.flatten()
    # y_star = y_star.flatten()
    # z_star = z_star.flatten()
    # t_star = t_star.flatten()
    
    x_test = x_star.reshape(x_star.shape[0], 1)
    y_test = y_star.reshape(y_star.shape[0], 1)
    z_test = z_star.reshape(z_star.shape[0], 1)
    t_test = t_star.reshape(t_star.shape[0], 1)
    
    u_pred, v_pred, w_pred, p_pred = model.predict(x_test, y_test, z_test, t_test, tensorfield)
    
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
    error_p = np.linalg.norm(p_star - p_pred, 2) / np.linalg.norm(p_star, 2)
    
    error_u_matrix = u_star - u_pred
    error_v_matrix = v_star - v_pred
    error_w_matrix = w_star - w_pred
    error_p_matrix = p_star - p_pred

    print('Error u: %e' % error_u)
    print('Error v: %e' % error_v)
    print('Error v: %e' % error_w)
    print('Error p: %e' % error_p)
    
    ######################################################################
    ############################## Plotting ##############################
    ######################################################################  
    v_mag = np.sqrt((u_star**2)+(v_star**2)+(w_star**2))
    v_pred_mag = np.sqrt((u_pred**2)+(v_pred**2)+(w_pred**2))
    pos_array = np.zeros((len(x_test), 2))
    pos_array[:,0] = x_test[:, 0]
    pos_array[:,1] = y_test[:, 0]
    pos_array = pos_array[pos_array[:, 0].argsort()]
    X, Y = np.meshgrid(pos_array[:,0], pos_array[:,1])
    
    def plot():
        # fig, ax = newfig(1.0, 1.2)
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.axis('off')
        
        ####### Row 0: h(t,x) ##################    
        gs0 = gridspec.GridSpec(1, 2)
        gs0.update(top=1-0.06, bottom=1-1/2 + 0.1, left=0.15, right=0.85, wspace=0)
        ax = plt.subplot(gs0[:, :])
        
        h = ax.imshow(uu.T, interpolation='nearest', cmap='seismic', 
                      extent=[tt.min(), tt.max(), x_star.min(), x_star.max()], 
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)
            
        idx_t0 = 20
        idx_t1 = 180
    
        line = np.linspace(xx.min(), xx.max(), 2)[:,None]
        ax.plot(tt[idx_t0]*np.ones((2,1)), line, 'w-', linewidth = 1)
        ax.plot(tt[idx_t1]*np.ones((2,1)), line, 'w-', linewidth = 1)
        
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        leg = ax.legend(frameon=False, loc = 'best')
        ax.set_title('$u(t,x)$', fontsize = 10)
        
        
        ####### Row 1: h(t,x) slices ##################    
        gs1 = gridspec.GridSpec(1, 2)
        gs1.update(top=1-1/2-0.05, bottom=0.15, left=0.15, right=0.85, wspace=0.5)
        
        ax = plt.subplot(gs1[0, 0])
        ax.plot(xx, uu[idx_t0,:], 'b-', linewidth = 2) 
        ax.plot(x_0, u_0, 'rx', linewidth = 2, label = 'Data')      
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title('$t = %.2f$' % (tt[idx_t0]), fontsize = 10)
        ax.set_xlim([b0-0.1, b1+0.1])
        ax.legend(loc='upper center', bbox_to_anchor=(0.8, -0.3), ncol=2, frameon=False)
    
    
        ax = plt.subplot(gs1[0, 1])
        ax.plot(xx, uu[idx_t1,:], 'b-', linewidth = 2, label = 'Exact') 
        ax.plot(x_star, u_pred[:,-1], 'r--', linewidth = 2, label = 'Prediction')      
        ax.set_xlabel('$x$')
        ax.set_ylabel('$u(t,x)$')    
        ax.set_title('$t = %.2f$' % (tt[idx_t1]), fontsize = 10)    
        ax.set_xlim([b0-0.1, b1+0.1])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.1, -0.3), ncol=2, frameon=False)
        
        plt.savefig('./figures/AC')
        plt.show()
        
    def plotPoints():
        #################### Training x,y #################### 
        # initial trainining data
        plt.plot(x_train, y_train, 'ob', label = 'Random', markersize=1)
        plt.plot(xb_train, yb_train, 'xr', label = 'Boundary', markersize=1)
        plt.title('Training Points')
        plt.legend(loc = 'upper right')
        plt.savefig('./figures/beltrami_3D/training_points.png', dpi=300, format='png')
        plt.show()
        
        plt.plot(x_train, y_train, 'ob', label = 'Random', markersize=1)
        plt.plot(xb_train, yb_train, 'xr', label = 'Boundary', markersize=1)
        plt.title('Training Points')
        plt.legend(loc = 'upper right')
        plt.savefig('./figures/beltrami_3D/training_points.png', dpi=300, format='png')
        plt.show()
        
        
        #################### Teste x,y #################### 
        fig, ax = plt.subplots(figsize = (10, 10))
        ax.plot(x_star, y_star, 'o', color='black');
        ax.set_title('Test Points')
        plt.savefig('./figures/beltrami_3D/test_points.png', dpi=300, format='png')
        plt.show()
        
        
        #################### Relacao pontos de treino e teste ####################
        quantidade = [len(x_train), len(xb_train), len(x_star)]
        pontos = ['T. Random', 'T. Borda', 'Teste']
        plt.bar(pontos, quantidade)
        plt.title('Training Points')
        plt.savefig('./figures/beltrami_3D/training_test_points_bars.png', dpi=300, format='png')
        plt.show()
    
    
    def plotVectors():
        #################### Teste x,y,u,v vectors #################### 
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.quiver(x_test, y_test, u_star, v_star, v_mag, cmap='jet')
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field')
        ## plt.colorbar()
        plt.savefig('./figures/beltrami_3D/velocities_exact_vectors.png', dpi=300, format='png')
        plt.show()
        
        
        #################### Teste x,y,u,v pred vectors ####################
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.quiver(x_test, y_test, u_pred, v_pred, v_pred_mag, cmap='jet')
        cbar = plt.colorbar(h, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field Pred')
        ## plt.colorbar()
        plt.savefig('./figures/beltrami_3D/velocities_pred_vectors.png', dpi=300, format='png')
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
        plt.savefig('./figures/beltrami_3D/velocities_exact_streamlines.png', dpi=300, format='png')
        plt.show()
        
    
    
        #################### Teste x,y,u,v pred streamlines #################### 
        fig, ax = plt.subplots(figsize = (10, 10))
        h = ax.streamplot(X, Y, VX_pred, VY_pred, density=1, linewidth=2, color = v_pred_mag, cmap='jet')
        cbar = plt.colorbar(h.lines, ax=ax)
        cbar.set_label("velocity mag", rotation=270, labelpad=20)
        ax.set_title('Vector Field Pred- Streamlines')
        plt.savefig('./figures/beltrami_3D/velocities_pred_streamlines.png', dpi=300, format='png')
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
        plot_solution(pos_array, p_star, 8, 500,'P exact','x','y','p')
        
        plot_solution(pos_array, error_u_matrix, 9, 500,'U exact - U predict','x','y','Ue - Up')
        plot_solution(pos_array, error_v_matrix, 10, 500,'V exact - V predict','x','y','Ve - Vp')
        plot_solution(pos_array, error_w_matrix, 11, 500,'W exact - W predict','x','y','We - Wp')
        plot_solution(pos_array, error_p_matrix, 12, 500,'P exact - P predict','x','y','Pe - Pp')
    
    # plot()
    # plotPoints()
    # plotVectors()
    plotStreamlines()
    # plotColormaps()
    # model.plotLoss()
    
