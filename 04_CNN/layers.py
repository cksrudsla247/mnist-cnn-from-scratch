# coding: utf-8
import numpy as np

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        
        return out
        
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # gradients of the weight and bias parameters
        self.dW = None
        self.db = None

    def forward(self, x):
        
        self.original_x_shape = x.shape
        # flatten a 4-D input to 2-D
        if x.ndim == 4:                    
            x = x.reshape(x.shape[0],-1)
        
        self.x = x
        out = np.dot(x,self.W) + self.b
        
        return out

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T)
        self.db  = np.sum(dout, axis = 0)
        
        # reshape dx back to the shape of x saved in forward
        return dx.reshape(*self.original_x_shape)

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # loss
        self.y = None    # softmax output
        self.t = None    # labels (one-hot)
        
    def forward(self, x, t):
        
        self.t = t
        
        c = np.max(x,axis = 1, keepdims = True)
        exp_a = np.exp(x-c)
        exp_sum = np.sum(exp_a, axis = 1, keepdims = True)
        self.y = exp_a/exp_sum
        
        if self.y.ndim == 1:
            self.y = self.y.reshape(1, self.y.size)
            self.t = self.t.reshape(1, self.t.size)
            
        batch_size = self.y.shape[0]
        
        self.loss = -np.sum(t * np.log(self.y + 1e-7)) / batch_size
        
        return self.loss
        
    def backward(self, dout=1):
        batch_size = self.y.shape[0]
        
        dx = (self.y-self.t)/batch_size
        return dx


class im2col_col2im:

    def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
        
        N, C, H, W = input_data.shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1

        img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
        return col
    
    def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):

        N, C, H, W = input_shape
        out_h = (H + 2*pad - filter_h)//stride + 1
        out_w = (W + 2*pad - filter_w)//stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]
    
    
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # intermediate data (used in backward)
        self.x = None   
        self.col = None
        self.col_W = None
        
        # gradients of the weight and bias parameters
        self.dW = None
        self.db = None
    
    def forward(self, x):
        # implemented below so that inference can run
        FN, _, FH, FW = self.W.shape
        N, _, H, W = x.shape
        
        self.x = x
        
        OH = int(((H+2*self.pad-FH)/self.stride) + 1)
        OW = int(((W+2*self.pad-FW)/self.stride) + 1)
        # four nested for loops (N, FN, OH, OW in that order)
        
        
        # forward with im2col
        col = im2col_col2im.im2col(x,FH,FW,self.stride, self.pad)
        col_W = self.W.reshape(FN,-1).T
        out = np.dot(col,col_W) + self.b
        
        out = out.reshape(N,OH,OW,-1).transpose(0,3,1,2)
        
        self.col = col
        self.col_W = col_W
        
        return out
        
        #forloop
        
        """
        x_pad = self.x
        
        if self.pad>0:
            x_pad = np.pad(self.x, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)),\
                           'constant', constant_values = 0)
        
        out = np.zeros((N,FN,OH,OW))

        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                        oh_start = oh*self.stride
                        oh_end = oh*self.stride+FH
                        ow_start = ow*self.stride
                        ow_end = ow*self.stride+FW
                        x_window = x_pad[n,:,oh_start:oh_end,ow_start:ow_end]
                        z = np.sum((x_window*self.W[fn]))
                        out[n,fn,oh,ow] = z
        
        out += self.b.reshape(1,FN,1,1)
        
        return out"""
        

    def backward(self, dout):
        N,C,H,W = self.x.shape
        FN,_,FH,FW = self.W.shape
        OH,OW = dout.shape[2],dout.shape[3]
        
        
        # backward with col2im 
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = im2col_col2im.col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
        #for loop
        """
        x_pad = self.x
        if self.pad > 0:
            x_pad = np.pad(self.x, ((0,0), (0,0), (self.pad, self.pad), (self.pad, self.pad)), 
                           'constant', constant_values=0)
        
        dx = np.zeros((N, C, H + 2*self.pad, W + 2*self.pad))
        self.dW = np.zeros((FN,C,FH,FW))

        #db 
        self.db = np.sum(dout, axis = (0,2,3)) 
        
        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                        
                        h_start = oh*self.stride
                        h_end = oh*self.stride+FH
                        W_start = ow*self.stride
                        W_end = ow*self.stride+FW
                        
                        dout_f = dout[n,fn,oh,ow]
                        
                        #dW
                        self.dW[fn] += (x_pad[n,:,h_start:h_end,W_start:W_end]*dout_f)
                        
                        #dx
                        dx[n,:,h_start:h_end,W_start:W_end] += dout_f*self.W[fn]
        
        return dx[:, :, self.pad:self.pad+H, self.pad:self.pad+W]
        """
                  
    
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        
        self.x = x
        
        N,FN,H,W = self.x.shape
        
        OH = int(((H+2*self.pad-self.pool_h)/self.stride) + 1)
        OW = int(((W+2*self.pad-self.pool_w)/self.stride) + 1)
        
        # forward with im2col
        col = im2col_col2im.im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        
        out = out.reshape(N, OH, OW, FN).transpose(0,3,1,2)
        
        self.x = x
        self.arg_max = arg_max
        
        return out
        
        
        #for loop
        """
        out = np.zeros((N,FN,OH,OW))
        self.arg_max = np.zeros((N,FN,OH,OW),dtype = int)
        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                        oh_start = oh*self.stride
                        oh_end = oh*self.stride+self.pool_h
                        ow_start = ow*self.stride
                        ow_end = ow*self.stride+self.pool_w
                        
                        x_window = self.x[n,fn,oh_start:oh_end,ow_start:ow_end]
                        out[n,fn,oh,ow] = np.max(x_window)
                        self.arg_max[n,fn,oh,ow] = np.argmax(x_window)
         
        return out    
        """   
        
        
    def backward(self, dout):
        
        N,FN,H,W = self.x.shape
        _,_,OH,OW = dout.shape

        
        # backward with col2im
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = im2col_col2im.col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
    
        #for loop
        """
        dpooling = np.zeros((N,FN,H,W))
        
        for n in range(N):
            for fn in range(FN):
                for oh in range(OH):
                    for ow in range(OW):
                    
                        oh_start = oh*self.stride
                        oh_end = oh*self.stride+self.pool_h
                        ow_start = ow*self.stride
                        ow_end = ow*self.stride+self.pool_w

                        h_argmax = (self.arg_max[n,fn,oh,ow]) // self.pool_w
                        w_argmax = (self.arg_max[n,fn,oh,ow]) % self.pool_w
                        dpooling_window = dpooling[n,fn,oh_start:oh_end,ow_start:ow_end]
                        dpooling_window[h_argmax,w_argmax] += dout[n,fn,oh,ow]
                        
        return dpooling
        """
      
class BatchNorm_MLP:
    def __init__(self, gamma = 1, beta = 0, eps = 1e-7):
        
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        
        self.dgamma = None
        self.dbeta = None
        
        self.x = None
        self.xmu = None
        self.ivar = None
        self.sqrtvar = None
        self.xhat = None
        
    def forward(self,x):
        self.x = x
        
        N = self.x.shape[0]
        
        mu = (1.0 / N) * np.sum(self.x, axis = 0)
        
        self.xmu = self.x - mu
        
        sq = self.xmu ** 2
        
        var = (1.0 / N) * np.sum(sq, axis = 0)
        
        self.sqrtvar = np.sqrt(var + self.eps)
        
        self.ivar = 1.0 / self.sqrtvar
        
        self.xhat = self.xmu * self.ivar
        
        gammax = self.gamma*self.xhat
        
        out = gammax + self.beta
        
        return out
    
    
    def backward(self, dout):
        
        N,D = self.x.shape
        #step 1
        self.dbeta = np.sum(dout, axis = 0)
        #step 2
        self.dgamma = np.sum(self.xhat * dout, axis = 0)
        dxhat = self.gamma * dout
        #step 3
        divar = np.sum(dxhat*self.xmu, axis = 0)
        dxmu1 = dxhat * self.ivar
        #step 4
        dsqrtvar = divar * (-1.0 / (self.sqrtvar ** 2))
        #step 5
        dvar = (0.5 / (self.sqrtvar)) * dsqrtvar
        #step 6
        dsq = (1.0/N) * np.ones((N,D)) * dvar
        #step 7
        dxmu2 = 2 * self.xmu * dsq
        #step 8
        dx1 = dxmu1 + dxmu2
        dmu = - np.sum(dxmu1+dxmu2,axis = 0)
        
        dx2 = (1.0/N) * np.ones((N,D)) * dmu
        
        dx = dx1 + dx2
        
        return dx
        
        
class BatchNorm_CNN:
    def __init__(self, gamma=1, beta=0, eps=1e-7, momentum=0.9):
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.dgamma = None
        self.dbeta = None

        self.x = None
        self.xmu = None
        self.ivar = None
        self.sqrtvar = None
        self.xhat = None

    def forward(self, x, train_flg=True):
        self.x = x
        N, C, H, W = x.shape
        x_2D = x.transpose(0, 2, 3, 1).reshape(-1, C)

        if self.running_mean is None:
            self.running_mean = np.zeros(C)
            self.running_var = np.zeros(C)

        if train_flg:
            mu = x_2D.mean(axis=0)
            self.xmu = x_2D - mu
            var = (self.xmu ** 2).mean(axis=0)
            self.sqrtvar = np.sqrt(var + self.eps)
            self.ivar = 1.0 / self.sqrtvar
            self.xhat = self.xmu * self.ivar

            self.running_mean = self.momentum*self.running_mean + (1-self.momentum)*mu
            self.running_var = self.momentum*self.running_var + (1-self.momentum)*var
        else:
            # inference : use the running statistics, batch size independent
            xc = x_2D - self.running_mean
            self.xhat = xc / np.sqrt(self.running_var + self.eps)

        out_2D = self.gamma*self.xhat + self.beta
        return out_2D.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, C, H, W = dout.shape
        dout_2D = dout.transpose(0, 2, 3, 1).reshape(-1, C)
        M = N*H*W

        self.dbeta = np.sum(dout_2D, axis=0)
        self.dgamma = np.sum(self.xhat * dout_2D, axis=0)
        dxhat = self.gamma * dout_2D

        divar = np.sum(dxhat*self.xmu, axis=0)
        dxmu1 = dxhat * self.ivar
        dsqrtvar = divar * (-1.0 / (self.sqrtvar**2))
        dvar = (0.5 / self.sqrtvar) * dsqrtvar
        dsq = (1.0/M) * np.ones((M, C)) * dvar
        dxmu2 = 2 * self.xmu * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -np.sum(dx1, axis=0)
        dx2 = (1.0/M) * np.ones((M, C)) * dmu

        dx_2D = dx1 + dx2
        return dx_2D.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        
    
        