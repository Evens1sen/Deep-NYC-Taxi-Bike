import sys
import os
import argparse
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
import time
import configparser
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import importlib.machinery, importlib.util

################# python input parameters #######################
parser = argparse.ArgumentParser()
parser.add_argument('-model',type=str,default='AGCRN',help='choose which model to train and test')
parser.add_argument('-version',type=int,default=0,help='train version')
parser.add_argument('-instep',type=int,default=12,help='input step')
parser.add_argument('-outstep',type=int,default=12,help='predict step')
parser.add_argument('-batch',type=int,default=64,help='batch size')
parser.add_argument('-epoch',type=int,default=200,help='training epochs')
parser.add_argument('-mode',type=str,default='train',help='train or eval')
parser.add_argument('-data',type=str,default='metrla',help='METR-LA or PEMS-BAY or PEMSD7M')
parser.add_argument('-debug',type=int,default=0,help='1:debug; 0: normal training and test')
parser.add_argument('cuda',type=int,default=3,help='cuda device number')
args = parser.parse_args() #python
# args = parser.parse_args(args=[])    # jupyter notebook
device = torch.device("cuda:{}".format(args.cuda)) if torch.cuda.is_available() else torch.device("cpu") 
################# data selection #######################
if args.data=='metrla':
    FLOWPATH = './data/METRLA/metr-la.h5'
#     ADJPATH = './data/METRLA/W_metrla.csv'
    DATANAME = 'METR-LA'
elif args.data=='pemsbay':
    a=1
################# Global Parameters setting #######################   
MODELNAME = args.model
BATCHSIZE = args.batch
EPOCH = args.epoch
if args.debug == 1 :
    EPOCH = 1
TIMESTEP_IN = args.instep
TIMESTEP_OUT = args.outstep
################# Statistic Parameters from init_config.ini #######################   
ini_config = configparser.ConfigParser()
ini_config.read('./init_config.ini',encoding='UTF-8')
config = ini_config[DATANAME]
N_NODE = int(config['N_NODE'])  # 207,325,228
CHANNEL = int(config['CHANNEL'])  # 1
LEARNING_RATE = float(config['LEARNING_RATE'])   # 0.001
PATIENCE = int(config['PATIENCE'])   # 10
OPTIMIZER = str(config['OPTIMIZER'])   # Adam
# OPTIMIZER = 'RMSprop' int(config['CHANNEL'])
# LOSS = 'MSE' int(config['CHANNEL'])
LOSS = str(config['LOSS'])   # MAE
TRAINRATIO = float(config['TRAINRATIO']) # 0.8
TRAINVALSPLIT = float(config['TRAINVALSPLIT'])  # 0.125  train:val:test = 7:1:2
################# random seed setting #######################
# torch.manual_seed(100)
# torch.cuda.manual_seed(100)
# np.random.seed(100)
# torch.backends.cudnn.deterministic = True
################# System Parameter Setting #######################
# KEYWORD = DATANAME + '_'+ args.model + '_in' + str(args.instep) + '_out' + str(args.outstep) +'_version_' + str(args.version)
# if args.mode == 'eval':
#     KEYWORD = DATANAME + args.model + '_in' + str(args.instep) + '_out' + str(args.outstep) +'_version_0to4' 
PATH = './save/' + DATANAME + '_'+ args.model + '_in' + str(args.instep) + '_out' + str(args.outstep) +'_version_'
single_version_PATH = PATH + str(args.version)
multi_version_PATH = PATH + '0to4'
import os
cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
##################  data preparation   #############################
data = pd.read_hdf(FLOWPATH) # [samples,nodes]
timestamp = data.index
timestamp = np.tile(timestamp, [data.shape[0],1])  # [samples,nodes]
data = data.values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)

def get_seq_timestamp(data,mode):
    data = pd.read_hdf(FLOWPATH) # [samples,nodes]
    timestamp = data.index
    timestamp = np.tile(timestamp, [data.shape[0],1])  # [samples,nodes]    
    seq_timestamp = seq(timestamp,mode='TEST')
    return seq_timestamp # [samples,time,nodes] [samples,T,N]   in test dataset

def seq(data, mode):
    # input data shape: [samples,nodes]
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    return XS, YS
    # output data shape: [batch, time, nodes, channel] [B,T,N,C]

def get_inputdata(mode, scaler, ifaddtime=False):
    data = pd.read_hdf(FLOWPATH) # [samples,nodes]
    stamp = data.index
    timestamp = np.tile(stamp, [data.shape[1],1]).transpose(1,0)  # [samples,nodes]
    data = data.values
    if scaler=='z_score':
        scaler = StandardScaler()
    elif scaler=='minmax':
        scaler = MinMaxScaler()
    sca_data = scaler.fit_transform(data)
    sca_seq_X, sca_seq_Y = seq(sca_data,mode)  # [samples,N]->[B,T,N,C]
    time_seq_X, time_seq_Y = seq(timestamp,mode) # [samples,N]->[B,T,N,C]
    if ifaddtime==True:
        timestamp = (timestamp - timestamp.astype("datetime64[D]")) / np.timedelta64(1, "D")
        sca_seq_timestamp_X, sca_seq_timestamp_Y = seq(timestamp,mode) # [samples,N]->[B,T,N,C]
        sca_seq_X = np.concatenate((sca_seq_X, sca_seq_timestamp_X), axis=-1)
    return sca_seq_X, sca_seq_Y
    # output data shape: [batch, time, nodes, channel] [B,T,N,C]
    
def getModel(name, device):
    ### load different baseline model.py  ###
    model_path = './model/' + args.model + '.py'   # AGCRN.py 的路径
    loader = importlib.machinery.SourceFileLoader('baseline_py_file', model_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    baseline_py_file = importlib.util.module_from_spec(spec)
    loader.exec_module(baseline_py_file)
########## select the baseline model ##########
    if args.model=='AGCRN':
        model = baseline_py_file.AGCRN(device, num_nodes=N_NODE, input_dim=CHANNEL,output_dim=CHANNEL, horizon=TIMESTEP_OUT).to(device)
    if args.model=='GWN':    
        ADJPATH = './data/METRLA/adj_mx.pkl'
        ADJTYPE = 'doubletransition'
        adj_mx = baseline_py_file.load_adj(ADJPATH,ADJTYPE)
        supports = [torch.tensor(i).to(device) for i in adj_mx]  
        model = baseline_py_file.GWN(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports).to(device)
###############################################
    ### initial the model parameters ###
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
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

def trainModel(name, device, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name, device)
    summary(model, (TIMESTEP_IN, N_NODE, CHANNEL), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    
    min_val_loss = np.inf
    wait = 0

    print('LOSS is :',LOSS)
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for epoch in range(EPOCH): # EPOCH
        starttime = datetime.now()     
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
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
            torch.save(model.state_dict(), single_version_PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(single_version_PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    torch_score = evaluateModel(model, criterion, train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS, YS_pred)
    with open(single_version_PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, device, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device)
    model.load_state_dict(torch.load(single_version_PATH+ '/' + name + '.pt'))
    
    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS)), scaler.inverse_transform(np.squeeze(YS_pred)) #[B,T,N,C]->[B,T,N]
    np.save(single_version_PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    np.save(single_version_PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f" % (name, mode, torch_score, torch_score))
    f = open(single_version_PATH + '/' + name + '_prediction_scores.txt', 'wt+')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
    print("3, 6, 12 pred steps evaluation: ")
    f.write("3, 6, 12 pred steps evaluation: ")
    for i in [2,5,11]:
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))        
    f.close()
    print('Model Testing Ended ...', time.ctime())

def multi_version_test(name, device, mode, XS, YS, versions):
    if LOSS == "MaskMAE":
        criterion = lib.Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('INPUT_STEP, PRED_STEP', TIMESTEP_IN, TIMESTEP_OUT)

    y_truth,y_pred=[],[]
    mse_all, rmse_all, mae_all, mape_all  = np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT)),np.zeros((len(versions),TIMESTEP_OUT))
    f = open(multi_version_PATH + '/' + name + '_multi_version_prediction_scores.txt', 'wt+')
    for v_ in versions:  
        XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
        test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
        test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
        model = getModel(name,device)       
        print('--- version ',v_,' evaluation start ---')
        f.write('--- version %s evaluation start ---' %(str(v_)))
        multi_test_PATH = './save/' + DATANAME + + args.model + '_in' + str(args.instep) + '_out' + str(args.outstep) +'_version_' + str(v_) + '/' + name + '.pt'   
        print('file path is :',multi_test_PATH)
        if os.path.isfile(multi_test_PATH):       
            model.load_state_dict(torch.load(multi_test_PATH))
            print("file exists")
        else:
            print("file not exist")
            break       
        YS_pred = predictModel(model, test_iter)
        YS_truth = YS
        YS_truth, YS_pred = scaler.inverse_transform(np.squeeze(YS_truth)), scaler.inverse_transform(np.squeeze(YS_pred))
#         YS_withtime = np.concatenate((YS_truth[:, :, :, np.newaxis].astype(str), time_seq_Y.astype(str)), axis=-1)  # [B,T,N,C], C:value+timestamp, save file as str type
#         YS_pred_withtime = np.concatenate((YS_pred[:, :, :, np.newaxis].astype(str) , time_seq_Y.astype(str)), axis=-1) # [B,T,N,C], C:value+timestamp, save file as str type    
        y_truth.append(YS_truth)
        y_pred.append(YS_pred)
        MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth, YS_pred)
        print('*' * 40)
        print(f'Version: {v_} Start Testing :')
        print("all pred steps in version %d, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (v_, name, mode, MSE, RMSE, MAE, MAPE))  
        for i in range(TIMESTEP_OUT):
            MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth[:, i, :], YS_pred[:, i, :])
            mse_all[v_,i], rmse_all[v_,i], mae_all[v_,i], mape_all[v_,i] = MSE, RMSE, MAE, MAPE
            print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE)) 
        print("3, 6, 12 pred steps evaluation: ")
        for i in [2,5,11]:
            MSE, RMSE, MAE, MAPE = lib.Metrics.evaluate(YS_truth[:, i, :], YS_pred[:, i, :])
            print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
            f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        print('*'*40)
        print('--- version ',v_,' evaluation end ---')
        f.write('--- version %s evaluation end ---' %(str(v_)))
        print('')
    y_truth=np.array(y_truth) #[versions,samples,time,nodes,channel] [V,samples,T,N]  
    y_pred=np.array(y_pred) #[versions,samples,time,nodes,channel] [V,samples,T,N] 
    print('y_truth.shape, y_pred.shape,', y_truth.shape, y_pred.shape)
    np.save(multi_version_PATH + '/' + MODELNAME + '_groundtruth.npy', y_truth)
    np.save(multi_version_PATH + '/' + MODELNAME + '_prediction.npy', y_pred)
    mse = np.array(mse_all).mean(axis=0)
    rmse = np.array(rmse_all).mean(axis=0)
    mae = np.array(mae_all).mean(axis=0)
    mape = np.array(mape_all).mean(axis=0)
    print('*'*40)
    print('*'*40)
    print('*'*40)
    print('Results in Test Dataset in Each Horizon with All Version Average:')    
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = mse[i], rmse[i], mae[i], mape[i]
        print("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE)) 
    print("Results in Test Dataset in 3, 6, 12 Horizon with All Version:")
    f.write("Results in Test Dataset in 3, 6, 12 Horizon with All Version: \n")
    for i in [2,5,11]:
        MSE, RMSE, MAE, MAPE = mse[i], rmse[i], mae[i], mape[i]
        print("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d Horizon, %s, %s, MSE, RMSE, MAE, MAPE, %.3f, %.3f, %.3f, %.3f\n" % (i+1, name, mode, MSE, RMSE, MAE, MAPE))    
    f.close()
    print('Model Multi Version Testing Ended ...', time.ctime())        
    

def main():
    if not os.path.exists(multi_version_PATH):
        os.makedirs(multi_version_PATH)
    if not os.path.exists(single_version_PATH):
        os.makedirs(single_version_PATH)
    model_path = './model/'+args.model+'.py'
    shutil.copy2(model_path, single_version_PATH)
    shutil.copy2(model_path, multi_version_PATH)
    
    if args.mode == 'train':    
        print(single_version_PATH, 'training started', time.ctime())
        trainXS, trainYS = get_inputdata(mode='TRAIN', scaler='z_score', ifaddtime=False)
        print('TRAIN XS.shape, YS,shape', trainXS.shape, trainYS.shape)
        trainModel(MODELNAME, device, 'train', trainXS, trainYS)

        print(single_version_PATH, 'testing started', time.ctime())
        testXS, testYS = get_inputdata(mode='TEST', scaler='z_score', ifaddtime=False)
        print('TEST XS.shape, YS.shape, time_seq_Y.shape :', testXS.shape, testYS.shape)
        testModel(MODELNAME, device, 'test', testXS, testYS)
    if args.mode == 'eval':
        print(multi_version_PATH, 'multi version testing started', time.ctime())
        testXS, testYS = get_inputdata(mode='TEST', scaler='z_score', ifaddtime=False)
        print('TEST XS.shape, YS.shape, time_seq_Y.shape :', testXS.shape, testYS.shape)
        multi_version_test(MODELNAME, device, args.mode, testXS, testYS, versions=np.arange(0,5))  #
    
    
if __name__ == '__main__':
    main()

