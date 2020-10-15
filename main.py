# coding: utf-8
#Main for the simulation
import os
import argparse
import torchvision.datasets as datasets
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
import pickle
import datetime
import numpy as np
import platform
import time

from tqdm import tqdm

from Tools import *
from Network import *
from plotFunction import*

parser = argparse.ArgumentParser(description='Quantized network with EP - binary weights & binary neurons - Full precision dynamics of pre-activations & sigma-delta modulation')
parser.add_argument(
    '--device',
    type=int,
    default=-1,
    help='GPU name to use cuda')
parser.add_argument(
    '--action',
    type=str,
    default='train_ep',
    help='train_ep: ep training, or test: test to initiate the network')
parser.add_argument(
    '--model',
    type=str,
    default='ave_prototypical',
    help='model for EP : prototypical or classic for energy based EP, to be trained: prototypical, ave_protoypical')
parser.add_argument(
    '--weightValue',
    type=str,
    default='method6',
    help='select value of weights in the network')
parser.add_argument(
    '--gradThreshold',
    nargs='+',
    type=float,
    default=[1e-6, 1e-6],
    help='threshold to flip the weights')
parser.add_argument(
    '--resetGrad',
    type=int,
    default=0,
    help='1: reset accGrad after a weight flip, 0: does not reset accGrad')
parser.add_argument(
    '--epochs',
    type=int,
    default=200,
    metavar='N',
help='number of epochs to train (default: 1)')
parser.add_argument(
    '--batchSize',
    type=int,
    default=64,
    help='Batch size (default=10)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=512,
    help='Testing Batch size (default=512)')
parser.add_argument(
    '--T',
    type=int,
    default=100,
    metavar='T',
    help='number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=100,
    metavar='Kmax',
    help='number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--beta',
    type=float,
    default=0.5,
    help='nudging parameter (default: 1)')
parser.add_argument(
    '--gamma',
    nargs='+',
    type=float,
    default=[2e-4, 5e-4],
    help='low-pass filter constant')
parser.add_argument(
    '--gamma_neur',
    type=float,
    default=5e-1,
    help='gamma to filter out pre-activations of neurons for relaxation')
parser.add_argument(
    '--clamped',
    type=int,
    default=1,
    help='Clamped neurons or not: crossed input are clamped to avoid divergence  (default: True)')
parser.add_argument(
    '--activationFun',
    type=str,
    default="heavyside",
    help='Binary activation function (: sign, heavyside)')
parser.add_argument(
    '--rho_threshold',
    type=float,
    default=0.5,
    help='threshold/offset of the activation function! 0.5 mean rho(x-0.5), 0 for rho(x)')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[784, 512, 10],
    help='List of layers in the model')
parser.add_argument(
    '--lrBias',
    nargs='+',
    type=float,
    default=[0.00005, 0.0001],
    help='learning rates for bias')
parser.add_argument(
    '--decay_thres',
    type=float,
    default=1.0,
    help='quantity by how much we decay the learning rate (and increase the threshold!)')
parser.add_argument(
    '--epoch_decay',
    type=float,
    default=20,
    help='nubmer of epochs between two adjustement on the learning rate')

args = parser.parse_args()


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes, activFun):
        self.number_classes = number_classes
        self.activFun = activFun

    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        if self.activFun == 'sign' or args.rho_threshold == 0.:
            target_onehot = -1*torch.ones((1,self.number_classes))
        else:
            target_onehot = torch.zeros((1,self.number_classes))
        return target_onehot.scatter_(1, target, 1).squeeze(0)


transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]


train_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=True, download=True,
                         transform=torchvision.transforms.Compose(transforms),
                         target_transform=ReshapeTransformTarget(10, args.activationFun)),
batch_size = args.batchSize, shuffle=True)

test_loader = torch.utils.data.DataLoader(
torchvision.datasets.MNIST(root='./data', train=False, download=True,
                         transform=torchvision.transforms.Compose(transforms),
                         target_transform=ReshapeTransformTarget(10, args.activationFun)),
batch_size = args.test_batchSize, shuffle=True)


def rho(x):
    # return (((torch.sign(x)+1)/2)*(x != 0).float() + (x == 0).float()).float()
    return (x > 0).float()
def rhop(x):
    #we use this convention as the x is pre-centered by args.rho_threshold before, so we say it is one if -0.5<x<0.5
    return ((x >= -0.5) & (x <= 0.5)).float()


if __name__ == '__main__':

    args.layersList.reverse() #we put in the other side, output first, input last

    net = Network(args)
    if net.cuda is True:
        net = net.to(net.device)
    args.action = "test"

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if args.action == 'test':
        print("Testing the model")

        data, target = next(iter(train_loader))
        s1, phi1 = net.initHidden(args, data)
        s1, phi1, y, h = net.forward(args, s1, phi1, tracking = True)
        seq1, phi1_seq = s1.copy(), phi1.copy()
        s1, phi1, y1, h1 = net.forward(args, s1, phi1, target = target, beta = net.beta, tracking = True)
        gradW1, gradBias1 = net.computeGradients(args, s1, phi1, seq1, phi1_seq)
        plt.plot(y+y1, label = 'output layer')
        plt.plot(h+h1, label = 'hidden layer')
        
        #we set the accumulated gradient to 0 in order to compare gradW1 and gradW2
        net.accGradients = []

        s2, phi2 = net.initHidden(args, data)
        s2, phi2, y, h = net.forward(args, s2, phi2, tracking = True)
        seq2, phi2_seq = s2.copy(), phi2.copy()
        s2, phi2, y1, h1 = net.forward(args, s2, phi2, target = target, beta = net.beta, tracking = True)
        gradW2, gradBias2 = net.computeGradients(args, s2, phi2, seq2, phi2_seq)

        plt.figure()
        plt.plot(y+y1, label = 'output layer')
        plt.plot(h+h1, label = 'hidden layer')
        plt.legend()
        plt.show()


    elif args.action == 'train_ep':
        print("Training the model")

        BASE_PATH, name = createPath(args)
        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net)

        train_error_list, test_error_list = [], []
        train_loss_list, test_loss_list = [], []

        for epoch in tqdm(range(args.epochs)):
            train_error, train_loss, nb_changes_epoch = train_bin(net, args, train_loader, epoch)
            test_error, test_loss = test_bin(net, args, test_loader)

            train_error_list.append(train_error.cpu().item())
            test_error_list.append(test_error.cpu().item())

            train_loss_list.append(train_loss.cpu().item())
            test_loss_list.append(test_loss.cpu().item())

            DATAFRAME = updateDataframe(BASE_PATH, args, DATAFRAME, net, train_error_list, test_error_list, train_loss_list, test_loss_list, nb_changes_epoch)
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')



# win: cd Desktop/Code/Fully-Quantized-Network-FP-SDM
# mac: cd Desktop/Thèse/Code/Fully-Quantized-Network-FP-SDM