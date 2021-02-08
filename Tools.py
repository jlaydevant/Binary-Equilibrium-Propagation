import os
import os.path
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import*
from copy import*
import sys
import pandas as pd
import shutil
from tqdm import tqdm
import glob


#================= FC ARCHITECTURES ===================================================
def train_fc(net, args, train_loader, epoch, optim = 'ep'):

    net.train()
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0
    nb_changes = [0. for k in range(len(args.layersList)-1)]

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        if args.random_beta == 1:
            net.beta = torch.sign(torch.randn(1)) * args.beta
            
        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        #free phase
        s = net.forward(s)
        seq = s.copy()

        #loss
        loss = (1/(2*s[0].size(0)))*criterion(s[0], targets) # loss for the mini-batch
        loss_loc += loss                                   # cumulative loss for the epoch

        #nudged phase
        s = net.forward(s, target = targets, beta = net.beta)

        #update and track the weights of the network
        nb_changes_loc = net.updateWeight(epoch, s, seq, args)
        for k in range(len(args.layersList)-1):
            nb_changes[k] = nb_changes[k] + nb_changes_loc[k]

        #compute error
        falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(seq[0], dim = 1)).int().sum(dim=0)

    train_error = (falsePred / float(len(train_loader.dataset))) * 100
    total_loss = loss_loc/ len(train_loader.dataset)
    
    return train_error, total_loss, nb_changes


def test_fc(net, args, test_loader):

    net.eval()
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0

    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        #free phase
        s = net.forward(s)

        #loss
        loss_loc += (1/(2*s[0].size(0)))*criterion(s[0], targets)

        #compute error
        falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(s[0], dim = 1)).int().sum(dim=0)

    test_error = (falsePred / float(len(test_loader.dataset))) * 100
    test_loss = loss_loc/ len(test_loader.dataset)

    return test_error, test_loss


def initDataframe_fc(path, args, net, dataframe_to_init = 'results.csv'):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Train_Error','Test_Error','Train_Loss','Test_Loss']

        for k in range(len(args.layersList) - 1):
            columns_header.append('Nb_Change_Weights_'+str(k))

        for k in range(len(net.threshold)):
            columns_header.append('Threshold_'+str(k))

        for k in range(len(args.gamma)):
            columns_header.append('Gamma_'+str(k))
            
        for k in range(len(net.weightOffset_tab)):
            columns_header.append('Alpha_'+str(k))
        
        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
    return dataframe


def updateDataframe_fc(BASE_PATH, args, dataframe, net, train_error_list, test_error_list, train_loss_list, test_loss_list, nb_changes_epoch):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    data = [train_error_list[-1], test_error_list[-1], train_loss_list[-1], test_loss_list[-1]]
    for k in range(len(args.layersList) - 1):
        N_weights = net.W[k].weight.numel()
        data.append(log(nb_changes_epoch[k]/N_weights+1e-9))

    for k in range(len(net.threshold)):
        data.append(net.threshold[k])

    for k in range(len(args.gamma)):
        data.append(args.gamma[k])
        
    for k in range(len(net.weightOffset_tab)):
        data.append(net.weightOffset_tab[k].cpu().item())

    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix + 'results.csv')

    return dataframe



#================= CONV ARCHITECTURES ===================================================
def train_conv(net, args, train_loader, epoch, optim = 'ep'):
    
    net.train()
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0
    nb_changes_fc = [0. for k in range(len(args.layersList)-1)]
    nb_changes_conv = [0. for k in range(len(args.convList)-1)]

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        if args.random_beta == 1:
            net.beta = torch.sign(torch.randn(1)) * args.beta

        s, inds = net.initHidden(args, data)
        
        if net.cuda:
            data, targets = data.to(net.device), targets.to(net.device)
            net.beta = net.beta.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        #free phase
        s, inds = net.forward(s, data, inds)

        seq = [item.clone() for item in s]
        indseq = [item if item is not None else None for item in inds ]

        #loss
        loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
        loss_loc += loss

        #nudged phase
        s, inds = net.forward(s, data, inds, beta = net.beta, target = targets, optim = optim)

        #update and track the weights of the network
        nb_changes_fc_loc, nb_changes_conv_loc = net.updateWeight(epoch, s, seq, inds, indseq, args, data, optim)

        nb_changes_fc   = [x1+x2 for (x1, x2) in zip(nb_changes_fc, nb_changes_fc_loc)]
        nb_changes_conv = [x1+x2 for (x1, x2) in zip(nb_changes_conv, nb_changes_conv_loc)]

        #compute error
        falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(seq[0], dim = 1)).int().sum(dim=0)

    train_error = (falsePred / float(len(train_loader.dataset))) * 100
    train_loss = loss_loc/ float(len(train_loader.dataset))

    return train_error, train_loss, nb_changes_fc, nb_changes_conv


def test_bin_conv(net, args, test_loader):

    net.eval()
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):

            s, inds = net.initHidden(args, data)

            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(len(s)):
                    s[i] = s[i].to(net.device)

            #free phase
            s, inds = net.forward(s, data, inds)

            #loss
            loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
            loss_loc += loss

            #compute error
            falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(s[0], dim = 1)).int().sum(dim=0)

        test_error = (falsePred / float(len(test_loader.dataset))) * 100
        test_loss = loss_loc/ len(test_loader.dataset)

    return test_error, test_loss


def initDataframe_conv(path, args, net, dataframe_to_init = 'results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        columns_header = ['Train_Error','Test_Error','Train_Loss','Test_Loss']

        for k in range(len(args.layersList) - 1):
            columns_header.append('Nb_Change_Weights_fc_'+str(k))

        for k in range(len(args.convList) - 1):
            columns_header.append('Nb_Change_Weights_conv_'+str(k))

        for k in range(len(net.fc_threshold)):
            columns_header.append('Threshold_fc_'+str(k))

        for k in range(len(net.conv_threshold)):
            columns_header.append('Threshold_conv_'+str(k))

        for k in range(len(args.classi_gamma)):
            columns_header.append('Gamma_fc_'+str(k))

        for k in range(len(args.conv_gamma)):
            columns_header.append('Gamma_conv_'+str(k))
        
        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
    return dataframe


def updateDataframe_conv(BASE_PATH, args, dataframe, net, train_error_list, test_error_list, train_loss_list, test_loss_list, nb_changes_epoch_fc, nb_changes_epoch_conv):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
        
    data = [train_error_list[-1], test_error_list[-1], train_loss_list, test_loss_list]

    for k in range(len(args.layersList) - 1):
        N_weights = net.fc[k].weight.numel()
        data.append(log(nb_changes_epoch_fc[k]/N_weights+1e-9))

    for k in range(len(args.convList) - 1):
        N_weights = net.conv[k].weight.numel()
        data.append(log(nb_changes_epoch_conv[k]/N_weights+1e-9))

    for k in range(len(net.fc_threshold)):
        data.append(net.fc_threshold[k])

    for k in range(len(net.conv_threshold)):
        data.append(net.conv_threshold[k])

    for k in range(len(args.classi_gamma)):
        data.append(args.conv_gamma[k])

    for k in range(len(args.conv_gamma)):
        data.append(args.conv_gamma[k])

    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix + 'results.csv')

    return dataframe


#================= COMMONS ===================================================


def createPath(args):
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        prefix = '\\'
        BASE_PATH = prefix + prefix + "?" + prefix + os.getcwd()
    else:
        prefix = '/'
        BASE_PATH = '' + os.getcwd()

    BASE_PATH += prefix + 'DATA-0-' + str(args.dataset)

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)

    filePath = shutil.copy('plotFunction.py', BASE_PATH)

    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            if names.split('.')[-1] != 'DS_Store':
                tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)

    os.mkdir(BASE_PATH)
    name = BASE_PATH.split(prefix)[-1]

    return BASE_PATH, name


def saveHyperparameters(args, net, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Binary-Weights Equilibrium Propagation \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write('\n')

    f.close()
