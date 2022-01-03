'''
GraphWaveNet.py
'''
import sys
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
import numpy as np
import pandas as pd
from torchsummary import summary
from Utils import load_pickle

import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
# parser.add_argument("--", type=, default=, help="")
parser.add_argument("--dataname", type=str, default="METR-LA", help="dataset name")
parser.add_argument("--timestep_in", type=int, default=12, help="the time step you input")
parser.add_argument("--timestep_out", type=int, default=3, help="the time step will output")
parser.add_argument("--n_node", type=int, default=69, help="the number of the node")
parser.add_argument("--channel", type=int, default=1, help="number of channel")
parser.add_argument("--batchsize", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--patience", type=float, default=10, help="patience used for early stop")
parser.add_argument("--optimizer", type=str, default='Adam', help="RMSprop, Adam")
parser.add_argument("--loss", type=str, default='MAE', help="MAE, MSE, SELF")
parser.add_argument("--trainratio", type=float, default=0.8,
                    help="the total ratio of training data and validation data")  # TRAIN + VAL
parser.add_argument("--trainvalsplit", type=float, default=0.125,
                    help="val_ratio = 0.8 * 0.125 = 0.1")  # val_ratio = 0.8 * 0.125 = 0.1
parser.add_argument("--flowpath", type=str, default='/home/cseadmin/mhy/data-NYCBike/60min/2019-2020-graph-outflow.npz',
                    help="the path of flow file")
parser.add_argument("--adjpath", type=str, default='/home/cseadmin/mhy/data-NYCZones/adjmatrix/W_od_bike_new.csv',
                    help="the path of adj file")
parser.add_argument("--target", type=int, default=0, help='the target dim')
parser.add_argument("--adjtype", type=str, default="normlap", help="the type of adj")
parser.add_argument('--ex', type=str, default='typhoon-inflow', help='which experiment setting to run')
parser.add_argument('--gpu', type=int, default=3, help='gpu num')
parser.add_argument('--addtime', type=bool, default=False, help="Add timestamp")


opt = parser.parse_args()

TARGET = opt.target
DATANAME = opt.dataname
TIMESTEP_OUT = opt.timestep_out
TIMESTEP_IN = opt.timestep_in
N_NODE = opt.n_node
CHANNEL = opt.channel
BATCHSIZE = opt.batchsize
LEARN = opt.lr
EPOCH = opt.epoch
PATIENCE = opt.patience
OPTIMIZER = opt.optimizer
LOSS = opt.loss
TRAINRATIO = opt.trainratio
TRAINVALSPLIT = opt.trainvalsplit
FLOWPATH = opt.flowpath
ADJPATH = opt.adjpath
ADJTYPE = opt.adjtype
GPU = opt.gpu
ADDTIME = opt.addtime


import os
cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)


ADJNUM = 2

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        x = torch.einsum('ncvl,vw->ncwl',(x,A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self,c_in,c_out):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=True)

    def forward(self,x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self,c_in,c_out,dropout,support_len=3,order=2):
        super(gcn,self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in,c_out)
        self.dropout = dropout
        self.order = order

    def forward(self,x,support):  # 这里是输入x，输入了support（图）
        out = [x]
        for a in support:
            x1 = self.nconv(x,a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1,a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out,dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        print(h.shape)
        return h   # 看下最后出来的维度


class gwnet(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.0, supports=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,out_dim=3,residual_channels=32,dilation_channels=32,skip_channels=256,end_channels=512,kernel_size=2,blocks=4,layers=2):
        super(gwnet, self).__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1,1))
        self.supports = supports

        receptive_field = 1

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device), requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device), requires_grad=True).to(device)
                self.supports_len +=1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1

        multi_gcn = nn.ModuleList()
        for i in range(ADJNUM):
            multi_gcn.append(gcn)
    
        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1,kernel_size),dilation=new_dilation))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *=2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(multi_gcn(dilation_channels,residual_channels,dropout,support_len=self.supports_len))  # gcn 改成multi_gcn


        
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                  out_channels=end_channels,
                                  kernel_size=(1,1),
                                  bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=out_dim,
                                    kernel_size=(1,1),
                                    bias=True)

        self.receptive_field = receptive_field



    def forward(self, input):
        in_len = input.size(3)
        if in_len<self.receptive_field:
            # print("forward start before", input.shape)
            x = nn.functional.pad(input,(self.receptive_field-in_len,0,0,0))
            # print("forward start after", x.shape)
        else:
            # print("forward start before", input.shape)
            x = input
            # print("forward start after", x.shape)

        # print("forward before", x.shape)
        x = self.start_conv(x)
        # print("forward after", x.shape)
        skip = 0

        # calculate the current adaptive adj matrix once per iteration
        
        # 这里需要修改一下，加个for循环 
        new_supports = None
        if self.gcn_bool and self.addaptadj and self.supports is not None:
            for j in range(ADJNUM):
                adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # adp搞成adp[j]
                new_supports = self.supports + [
                    adp]  # new_supports也搞成new_supports[j]   self.supports 对应是要改成 self.supports[j]  [adp[j]]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)
            residual = x
            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            # print("filter before:",x.shape)
            #todo torch.Size([2, 32, 69, 3]) ---> filter after: torch.Size([2, 32, 69, 1])
            x = filter * gate
            # print("filter after:",x.shape)

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # print("gconv before:", x.shape)
            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)  #    new_supports也搞成new_supports[j]   gconv[i]改成gconv[i][j]
                else:
                    x = self.gconv[i](x,self.supports)  #  self.supports也搞成self.supports[j]   gconv[i]改成gconv[i][j]
            else:   
                x = self.residual_convs[i](x)

            # print("gconv after:",x.shape)
            x = x + residual[:, :, :, -x.size(3):]
            # print("residual after",x.shape)

            x = self.bn[i](x)
            # print("bn after",x.shape)

        # print("F.relu before",x.shape)
        x = F.relu(skip)
        # print("before: ",x.shape)
        x = F.relu(self.end_conv_1(x))
        # print("after: ",x.shape)
        x = self.end_conv_2(x)
        # print("after2: ",x.shape)
        return x
    
def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_adj(pkl_filename, adjtype):
#**************    
#change adj input
#     sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)    

    adj_mx = pd.read_csv(pkl_filename).values
    distances = adj_mx[~np.isinf(adj_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(adj_mx / std))    
#**************       
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return adj

def main():
    # GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
    device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
    
    # adj_mx = [0]* 多图个数j
    # adj_mx = [1][2]
    adj_mx =load_adj(ADJPATH,ADJTYPE)  # for in range(多图个数j): adj_mx[j] = load_adj(ADJPATH,ADJTYPE)
    # supports = [0]* 多图个数j
    # supports = [1][2]
    supports = [torch.tensor(i).to(device) for i in adj_mx]  # supports[j] = [torch.tensor(i).to(device) for i in adj_mx[j]]   -> j 1 N N 

    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports).to(device)
    summary(model, (CHANNEL, N_NODE, TIMESTEP_IN), device=device)
    
if __name__ == '__main__':
    main()
