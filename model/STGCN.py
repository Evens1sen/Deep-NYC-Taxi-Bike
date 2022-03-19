'''
STGCN
'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigs
from torchsummary import summary

'''
align主要是对数据格式进行一个处理，类似于reshape
'''

class align(nn.Module):
    def __init__(self, c_in, c_out):
        super(align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)

    def forward(self, x):
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x
'''
门控卷积单元的定义，目的是用来提取时空特征
GLU和sigmoid
'''
class temporal_conv_layer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(temporal_conv_layer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        x_in = self.align(x)[:, :, self.kt - 1:, :]
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)
        return torch.relu(self.conv(x) + x_in)

    '''
    门控卷积单元的定义，目的是用来提取时空特征
    GLU和sigmoid
    '''

    class temporal_conv_layer(nn.Module):
        def __init__(self, kt, c_in, c_out, act="relu"):
            super(temporal_conv_layer, self).__init__()
            self.kt = kt
            self.act = act
            self.c_out = c_out
            self.align = align(c_in, c_out)
            if self.act == "GLU":
                self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
            else:
                self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

        def forward(self, x):
            x_in = self.align(x)[:, :, self.kt - 1:, :]
            if self.act == "GLU":
                x_conv = self.conv(x)
                return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            if self.act == "sigmoid":
                return torch.sigmoid(self.conv(x) + x_in)
            return torch.relu(self.conv(x) + x_in)
'''
空间卷积层
'''

class spatio_conv_layer(nn.Module):
    def __init__(self, ks, c, Lk):
        super(spatio_conv_layer, self).__init__()
        self.Lk = Lk
        self.theta = nn.Parameter(torch.FloatTensor(c, c, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        # print('spatio_conv_layer Lk.shape' + str(self.Lk.shape))
        # print('spatio_conv_layer x.shape' + str(x.shape))
        
        x_c = torch.einsum("knm,bitm->bitkn", self.Lk, x)
        # print('x_c shape is', x_c.shape)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b
        
        # print('sp shape is :', x_gc.shape)
        return torch.relu(x_gc + x)

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

'''
这就是对应的一个ST_conv模块，会调用到前面定义好的tconv和sconv
'''

class st_conv_block(nn.Module):
    # self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
    def __init__(self, ks, kt, n, c, p, Lk):
        super(st_conv_block, self).__init__()
        self.tconv1 = temporal_conv_layer(kt, c[0], c[1], "GLU")
        self.sconvlist = nn.ModuleList()
        self.sconvseq = nn.Sequential()
        for i in range(len(Lk)):
            self.sconvseq.add_module('spatio_conv_layer', spatio_conv_layer(ks, c[1], Lk[i]))
        # print(len(Lk))
        # print('c is', c)
        for i in range(len(Lk)):
            self.sconvlist.append(spatio_conv_layer(ks, c[1], Lk[i]))
        self.sconvtest = spatio_conv_layer(ks, c[1], Lk[0])
        self.sconv = spatio_conv_layer(ks, c[1], Lk)
        self.mlp = linear(c[1]*len(Lk), c[1])
        # def __init__(self, kt, c_in, c_out, act="relu"):
        self.tconv2 = temporal_conv_layer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)
        self.Lk = Lk

    def forward(self, x):
        x_t1 = self.tconv1(x)
        # print('x_t1 shape is',x_t1.shape)
        # x_s = self.sconvlist(x_t1)
        # x_s = self.sconvseq(x_t1)
        # x_s = torch.empty(len(self.Lk), x_t1.shape).to(device)
        # print('x_s shape before', x_s.shape)
        # x_s = torch.zeros_like(x_t1)
        x_s = torch.empty_like(x_t1)
        for i,layer in enumerate(self.sconvlist):
            # x_s += layer(x_t1)
            # print(i)
            if i == 0:
                x_s = layer(x_t1)
            else:
                x_s = torch.cat((x_s, layer(x_t1)), 1)
        # print('x_s shape before mlp', x_s.shape)
        x_s = self.mlp(x_s)
        # print('x_s shape after mlp', x_s.shape)
        # x_s = x_t1
        # x_s = self.sconvtest(x_t1)
        # x_s = self.sconv(x_t1)
        # print('st_conv_block x_s shape is',x_s.shape)
        x_t2 = self.tconv2(x_s)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)        
        return self.dropout(x_ln)

class output_layer(nn.Module):
    def __init__(self, c, T, n):
        super(output_layer, self).__init__()
        self.tconv1 = temporal_conv_layer(T, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = temporal_conv_layer(1, c, c, "sigmoid")
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_t2 = self.tconv2(x_ln)
        return self.fc(x_t2)

class STGCN(nn.Module):
    # model = STGCN(ks, kt, bs, T, n, Lk, p).to(device)
    # bs = [[CHANNEL, 16, 64], [64, 16, 64]]
    def __init__(self, ks, kt, bs, T, n, Lk, p):
        super(STGCN, self).__init__()
        self.st_conv1 = st_conv_block(ks, kt, n, bs[0], p, Lk)
        self.st_conv2 = st_conv_block(ks, kt, n, bs[1], p, Lk)
        self.output = output_layer(bs[1][2], T - 4 * (kt - 1), n)

    def forward(self, x):
        # print('x.shape is', x.shape)
        x_st1 = self.st_conv1(x)
        # print('x_st1.shape is ', x_st1.shape)
        x_st2 = self.st_conv2(x_st1)
        # print('x_st2.shape is ', x_st2.shape)
        output_data = self.output(x_st2)
        # print('output.shape is', output_data.shape)
        return output_data

def weight_matrix(W, sigma2=0.1, epsilon=0.5, alpha=10):
    '''
    :param sigma2: float, scalar of matrix W.
    :param epsilon: float, thresholds to control the sparsity of matrix W.
    :param scaling: bool, whether applies numerical scaling on W.
    :return: np.ndarray, [n_route, n_route].
    '''
    n = W.shape[0]
    W = W /alpha
    W[W==0]=np.inf
    W2 = W * W
    W_mask = (np.ones([n, n]) - np.identity(n))
    return np.exp(-W2 / sigma2) * (np.exp(-W2 / sigma2) >= epsilon) * W_mask

def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)

def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)

def main():
    from pred_STGCN import CHANNEL, N_NODE, TIMESTEP_IN, TIMESTEP_OUT, ADJPATH, ADJTYPE
    GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    print('STGCN.py ' + str(CHANNEL))
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 32, 64], [64, 32, 128]], TIMESTEP_IN, N_NODE, 0
    ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 32, 64], [64, 32, 64]], TIMESTEP_IN, N_NODE, 0
    # ks, kt, bs, T, n, p = 3, 3, [[CHANNEL, 16, 64], [64, 16, 64]], TIMESTEP_IN, N_NODE, 0
    Lk_new = torch.empty(2,3,69,69).to(device)
    for i in range(2):
        A = pd.read_csv(ADJPATH).values
        W = weight_matrix(A, 0.5, 0.5, 10)
        L = scaled_laplacian(W)
        # print('L shape is', L.shape)
        Lk = cheb_poly(L, ks)
        # print('cheb_poly', Lk.shape)
        Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
        Lk_new[i] = Lk
        # print('Lk shape is',Lk.shape)
    # print('Lk_new shape is',Lk_new.shape)
    # A = pd.read_csv(ADJPATH).values
    # W = weight_matrix(A)
    # L = scaled_laplacian(W)
    # print('L shape is', L.shape)
    # Lk = cheb_poly(L, ks)
    # print('cheb_poly', Lk.shape)
    # Lk = torch.Tensor(Lk.astype(np.float32)).to(device)
    # # # Lk_new = np.append(Lk_new, Lk, axis=0)
    # print('Lk shape is', Lk.shape)
    # model = STGCN(ks, kt, bs, T, n, Lk, p).to(device)
    model = STGCN(ks, kt, bs, T, n, Lk_new, p).to(device)
    summary(model, (CHANNEL, TIMESTEP_IN, N_NODE), device=device)
    torch.save(model, '../diagram/STGCN.pth')
    
if __name__ == '__main__':
    main()
