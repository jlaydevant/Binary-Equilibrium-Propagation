# coding: utf-8

import os
import os.path
import datetime
import time
import numpy as np
import torch.nn as nn
from scipy import*
from copy import*
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import shutil

from Network import*


def train_bin(net, args, train_loader, epoch, trackChanges = False):
    '''
    Function to train the network for 1 epoch
    '''
    net.train()
    net.epoch = epoch+1
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0
    nb_changes = [0. for k in range(len(args.layersList)-1)]

    #learning rate decay
    if epoch % args.epoch_decay == 0 and epoch != 0:
        for k in range(len(args.lrBias)):
            args.lrBias[k] = args.lrBias[k]/args.decay_thres
            net.threshold[k] = net.threshold[k]*args.decay_thres

    for batch_idx, (data, targets) in enumerate(train_loader):

        net.beta = torch.sign(torch.randn(1)) * args.beta

        s, phi = net.initHidden(args, data)

        if net.cuda:
            #no need to put data on the GPU as data is included in s!
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)
            for i in range(len(phi)):
                phi[i] = phi[i].to(net.device)
                

        #free phase
        s, phi = net.forward(args, s, phi)

        seq, phi_seq = s.copy(), phi.copy()

        #loss
        loss = (1/2)*criterion(s[0], targets*net.neuronMax) # loss for the mini-batch
        loss_loc += loss                                   # cumulative loss for the epoq

        #nudged phase
        s, phi = net.forward(args, s, phi, target = targets, beta = net.beta)

        #update and track the weights of the network
        nb_changes_loc = net.updateWeight(epoch, s, phi, seq, phi_seq, args)

        if trackGrad is None:
            for k in range(len(args.layersList)-1):
                nb_changes[k] = nb_changes[k] + nb_changes_loc[k]

        else:
            trackGrad, trackGradFilt, trackW = trackMulti(net, trackGrad, trackGradFilt, trackW)

        #compute error
        falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(seq[0], dim = 1)).int().sum(dim=0)

        del s, seq, phi, data, targets

    train_error = (falsePred.float() / float(len(train_loader.dataset))) * 100
    total_loss = loss_loc/ float(len(train_loader.dataset))

    if trackChanges:
        return nb_changes, train_error

    else:
        return train_error, total_loss, nb_changes


def test_bin(net, args, test_loader, method = 'train_ep'):
    '''
    Function to test the network
    '''
    net.eval()
    criterion = nn.MSELoss(reduction = 'sum')
    falsePred, loss_loc = 0, 0

    for batch_idx, (data, targets) in enumerate(test_loader):

        s, phi = net.initHidden(args, data)

        if net.cuda:
            targets = targets.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)
            for i in range(len(phi)):
                phi[i] = phi[i].to(net.device)

        #free phase
        s, _ = net.forward(args, s, phi)

        #loss
        loss_loc += (1/(2*s[0].size(0)))*criterion(s[0], targets*net.neuronMax)

        #compute error
        falsePred += (torch.argmax(targets, dim = 1) != torch.argmax(s[0], dim = 1)).int().sum(dim=0)

    test_error = (falsePred.float() / float(len(test_loader.dataset))) * 100
    test_loss = loss_loc
    # print('test error = '+str(test_error))

    return test_error, test_loss


def initDataframe(path, args, net, dataframe_to_init = 'results.csv'):
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
        columns_header = ['Train_Error','Test_Error','Min_Train_Error','Min_Test_Error', 'Train_Loss','Test_Loss' ]

        for k in range(len(args.layersList) - 1):
            columns_header.append('Nb_Change_Weights_'+str(k))

        for k in range(len(net.threshold)):
            columns_header.append('Threshold_'+str(k))

        for k in range(len(args.gamma)):
            columns_header.append('Gamma_'+str(k))

        dataframe = pd.DataFrame({},columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
    return dataframe


def updateDataframe(BASE_PATH, args, dataframe, net, train_error_list, test_error_list, train_loss_list, test_loss_list, nb_changes_epoch):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [train_error_list[-1], test_error_list[-1], min(train_error_list), min(test_error_list), train_loss_list[-1], test_loss_list[-1]]
    for k in range(len(args.layersList) - 1):
        N_weights = net.W[k].weight.numel()
        data.append(log(nb_changes_epoch[k]/N_weights+1e-9))

    for k in range(len(net.threshold)):
        data.append(net.threshold[k])

    for k in range(len(args.gamma)):
        data.append(args.gamma[k])

    new_data = pd.DataFrame([data],index=[1],columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(BASE_PATH + prefix +'results.csv')

    return dataframe


def createPath(args):
    '''
    Create path to save data
    '''

    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH += prefix + 'DATA-0'

    BASE_PATH += prefix + args.action

    BASE_PATH += prefix + args.model
    BASE_PATH += prefix + 'activFun-' + args.activationFun

    BASE_PATH += prefix + args.weightValue
    BASE_PATH += prefix + 'thres-' + str(args.gradThreshold)

    BASE_PATH += prefix + str(len(args.layersList)-2) + 'hidden'
    BASE_PATH += prefix + 'hidNeu' + str(args.layersList[1])

    BASE_PATH += prefix + 'β-' + str(args.beta)
    BASE_PATH += prefix + 'γ_Neur-' + str(args.gamma_neur)
    BASE_PATH += prefix + 'T-' + str(args.T)
    BASE_PATH += prefix + 'K-' + str(args.Kmax)

    BASE_PATH += prefix + 'Clamped-' + str(bool(args.clamped))[0]
    BASE_PATH += prefix + 'γ-' + str(args.gamma)

    BASE_PATH += prefix + 'lrB-' + str(args.lrBias)

    BASE_PATH += prefix + 'BaSize-' + str(args.batchSize)

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
        if '.DS_Store' in files:
            files.pop(files.index('.DS_Store'))
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)

    try:
        os.mkdir(BASE_PATH)
    except:
        pass
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
    if args.action == 'train_bpptt':
        f.write('Binary BPTT training \n')
    else:
        f.write('Binary Equilibrium Propagation \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        if key == "gradThreshold":
            f.write(str(net.threshold))
        else:
            f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()
