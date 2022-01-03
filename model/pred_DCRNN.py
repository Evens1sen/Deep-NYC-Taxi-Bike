import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import Metrics
import Utils
from DCRNN import *
# from Param import *
# from Param_GraphWaveNet import *
import argparse
from configparser import ConfigParser

parser = argparse.ArgumentParser()
# parser.add_argument("--", type=, default=, help="")
parser.add_argument("--dataname", type=str, default="NYC-Taxi", help="dataset name")
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
parser.add_argument("--flowpath", type=str, default='../data-NYCBike/60min/2019-2020-graph-inflow.npz', help="the path of flow file")
parser.add_argument("--adjpath", type=str, default='../data-NYCZones/adjmatrix/W_od_bike_new.csv', help="the path of adj file")
parser.add_argument("--cpu", type=int, default=1, help="the number of cpu")
parser.add_argument("--adjtype", type=str, default="normlap", help="the type of adj")
parser.add_argument('--ex', type=str, default='typhoon-inflow', help='which experiment setting to run')
parser.add_argument('--gpu', type=int, default=0, help='gpu num')
parser.add_argument('--target', type=int, default=0, help="The output target dimension")
parser.add_argument('--addtime', type=bool, default=False, help="Add timestamp")

opt = parser.parse_args()

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
TARGET = opt.target
ADDTIME = opt.addtime
cpu_num = opt.cpu  # cpu_num = 1
import os

os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN, data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i + TIMESTEP_IN, :]
            y = data[i + TIMESTEP_IN:i + TIMESTEP_IN + TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    #todo
    if CHANNEL ==1 :
        XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    else: 
        YS = YS[:, :, :, TARGET]
        YS = YS[:, :, :, np.newaxis]
    
    return XS, YS

def getXSYSTimestamp(data, timestamp, mode):
    global CHANNEL
    if ADDTIME:
        CHANNEL = 1
    XS, YS = getXSYS(data,mode)  # [samples,N]->[B,T,N,C]
    time_seq_X, time_seq_Y = getXSYS(timestamp,mode) # [samples,N]->[B,T,N,C]
    timestamp = (timestamp - timestamp.astype("datetime64[D]")) / np.timedelta64(1, "D")
    sca_seq_timestamp_X, sca_seq_timestamp_Y = getXSYS(timestamp,mode) # [samples,N]->[B,T,N,C]
    XS = np.concatenate((XS, sca_seq_timestamp_X), axis=-1)
    return XS, YS
    

def getModel(name):
    adj_mx = load_adj(ADJPATH, ADJTYPE)
    model = DCRNN(device, num_nodes=N_NODE, input_dim=CHANNEL, output_dim=1, out_horizon=TIMESTEP_OUT, P=adj_mx).to(device)
    return model


def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred


def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    global CHANNEL
    if ADDTIME:
        CHANNEL = 2
    model = getModel(name)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)

    min_val_loss = np.inf
    wait = 0

    print('LOSS is :', LOSS)
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            # print('y.shape: ', y.shape)
            # print('y_pred.shape: ', y_pred.shape)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, "validation loss:",
              val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
            "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))

    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    # todo
    # for dim in range(CHANNEL):
    #     YS[:, :, :, dim] = scaler.inverse_transform(YS[:, :, :, dim])
        # YS_pred[:, :, :, dim] = scaler.inverse_transform(YS_pred[:, :, :, dim])
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())


def testModel(name, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    global CHANNEL
    if ADDTIME:
        CHANNEL = 2
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))

    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    # if CHANNEL == 1:
    #     YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    # else:
    #     for dim in range(CHANNEL):
    #         YS[:, :, :, dim] = scaler.inverse_transform(YS[:, :, :, dim])
    #         YS_pred[:, :, :, dim] = scaler.inverse_transform(YS_pred[:, :, :, dim])
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print(
        "all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
    name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (
        i + 1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
        i + 1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'DCRNN'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
###########################################################
# GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################
# data = pd.read_hdf(FLOWPATH).values
data = None
timestamp = None
if ADDTIME:
    CHANNEL = 1
    data = pd.read_hdf(FLOWPATH)
    stamp = data.index
    timestamp = np.tile(stamp, [data.shape[1],1]).transpose(1,0)  # [samples,nodes]
    data = data.values
else:
    data = np.load(FLOWPATH)['arr_0']

scaler = StandardScaler()
if CHANNEL == 1:
    data = scaler.fit_transform(data)
else:
    for dim in range(CHANNEL):
        if dim == TARGET:
            continue
        data[:, :, dim] = scaler.fit_transform(data[:, :, dim])
    data[:, :, TARGET] = scaler.fit_transform(data[:, :, TARGET])
print('data.shape', data.shape)
###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('DCRNN.py', PATH)
    # shutil.copy2('Param.py', PATH)
    # shutil.copy2('Param_DCRNN.py', PATH)
    print(KEYWORD, 'training started', time.ctime())
    if ADDTIME:
        trainXS, trainYS = getXSYSTimestamp(data, timestamp, 'TRAIN')
    else:
        trainXS, trainYS = getXSYS(data, 'TRAIN')        
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)

    print(KEYWORD, 'testing started', time.ctime())
    if ADDTIME:
        testXS, testYS = getXSYSTimestamp(data, timestamp, 'TEST')
    else:
        testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)


if __name__ == '__main__':
    main()
