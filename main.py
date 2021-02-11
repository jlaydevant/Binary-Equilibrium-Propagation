import os
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F
import datetime
import numpy as np
import platform
import time
from tqdm import tqdm

from Tools import *
from Network import *


parser = argparse.ArgumentParser(description='Binarized weights version of EqProp')
#hardware settings
parser.add_argument(
    '--device',
    type=int,
    default=-1,
    help='GPU name to use cuda (default = -1 (no GPU) - other: 0, 1, ...')
#Architecture settings
parser.add_argument(
    '--archi',
    type=str,
    default='conv',
    help='Architecture to be trained: (default = "fc", other: "conv")')
parser.add_argument(
    '--layersList',
    nargs='+',
    type=int,
    default=[10],
    help='List of layer sizes (default: 10)')
parser.add_argument(
    '--convList',
    nargs='+',
    type=int,
    default=[64, 32, 1],
    help='List of channels for the convolution part (default = [64, 32, 10])')
parser.add_argument(
    '--padding',
    type=int,
    default=0,
    metavar='P',
    help='Padding (default: 0)')
parser.add_argument(
    '--kernelSize',
    type=int,
    default=5,
    metavar='KSize',
    help='Kernel size for convolution (default: 5)')
parser.add_argument(
    '--Fpool',
    type=int,
    default=2,
    metavar='Fp',
    help='Pooling filter size (default: 2)')
#EqProp settings
parser.add_argument(
    '--optim',
    type=str,
    default="train_ep",
    help='Optimiser to use: EP of BPTT (default: ep, others: bptt)')
parser.add_argument(
    '--activationFun',
    type=str,
    default="hardsigm",
    help='Activation function (default: hardsigm, others: sigm, tanh)')
parser.add_argument(
    '--T',
    type=int,
    default=150,
    metavar='T',
    help='Number of time steps in the free phase (default: 50)')
parser.add_argument(
    '--Kmax',
    type=int,
    default=10,
    metavar='Kmax',
    help='Number of time steps in the backward pass (default: 10)')
parser.add_argument(
    '--gamma_neur',
    type=float,
    default=5e-1,
    help='gamma to filter out pre-activations of neurons for relaxation')
parser.add_argument(
    '--beta',
    type=float,
    default=0.3,
    help='nudging parameter (default: 1)')
parser.add_argument(
    '--random_beta',
    type=int,
    default=1,
    help='Use random sign of beta for training or fixed >0 sign (default: 1, other: 0)')
#Binary optimization settings
parser.add_argument(
    '--binary_settings',
    type=str,
    default='bin_W',
    help='Binary version of EP to use: binary weights only or binary weights + binary activation (default=bin_W, other: bin_W_N')
parser.add_argument(
    '--gamma',
    nargs='+',
    type=float,
    default=[1e-5, 1e-5],
    help='Low-pass filter constant')
parser.add_argument(
    '--classi_gamma',
    nargs='+',
    type=float,
    default=[1e-5],
    help='Low-pass filter constant for the classifier for conv archi (default: [1], for 1 layer classifier)')
parser.add_argument(
    '--conv_gamma',
    nargs='+',
    type=float,
    default=[1e-5, 1e-5],
    help='Low-pass filter constant for the convolution part (default: [1, 1], for 2 convolution layers )')
parser.add_argument(
    '--gradThreshold',
    nargs='+',
    type=float,
    default=[5e-7, 5e-8],
    help='Thresholds used for the binary optimization')
parser.add_argument(
    '--classi_threshold',
    nargs='+',
    type=float,
    default=[5e-8],
    help='Thresholds used for the classifier of the conv archi')
parser.add_argument(
    '--conv_threshold',
    nargs='+',
    type=float,
    default=[5e-8, 5e-8],
    help='Thresholds used for the conv partd of the conv archi') 
parser.add_argument(
    '--rho_threshold',
    type=float,
    default=0.5,
    help='Offset of the activation function - 0.5 mean rho(x-0.5), 0 for rho(x)')
parser.add_argument(
    '--expand_output',
    type=int,
    default=1,
    help='Quantity by how much we enlarge the output layer (default=1)')
#Training settings
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='dataset used to train the network (default=mnist, other: cifar10')
parser.add_argument(
    '--lrBias',
    nargs='+',
    type=float,
    default=[0.025, 0.05, 0.1],
    help='learning rates for bias')
parser.add_argument(
    '--batchSize',
    type=int,
    default=64,
    help='Batch size (default=10)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=512,
    help='Testing B0atch size (default=512)')
parser.add_argument(
    '--epochs',
    type=int,
    default=2,
    metavar='N',
help='number of epochs to train (default: 1)')
#Learning the scaling factor
parser.add_argument(
    '--learnAlpha',
    type=int,
    default=1,
    help='Learn the scaling factors or let them fixed (default: 1, other: 0)')
parser.add_argument(
    '--lrAlpha',
    nargs='+',
    type=float,
    default=[0, 0, 0],
    help='learning rates for the scaling factors')
parser.add_argument(
    '--decayLrAlpha',
    type=float,
    default=1.0,
    help='Quantity by how much we decay the learning rate for alpha')
parser.add_argument(
    '--epochDecayLrAlpha',
    type=int,
    default=1000,
    help='Epoch to decay learning rate for alpha')
args = parser.parse_args()



class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


class ReshapeTransformTarget:
    def __init__(self, number_classes):
        self.number_classes = number_classes

    def __call__(self, target):
        target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
        target_onehot = torch.zeros((1,self.number_classes))
        return target_onehot.scatter_(1, target, 1).repeat_interleave(args.expand_output).squeeze(0)

if args.archi == "fc":    
    transforms=[torchvision.transforms.ToTensor(),ReshapeTransform((-1,))]
    
    train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=True, download=True,
                            transform=torchvision.transforms.Compose(transforms),
                            target_transform=ReshapeTransformTarget(10)),
    batch_size = args.batchSize, shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root='./data', train=False, download=True,
                            transform=torchvision.transforms.Compose(transforms),
                            target_transform=ReshapeTransformTarget(10)),
    batch_size = args.test_batchSize, shuffle=True)
    
    
elif args.archi == "conv":
    if args.dataset == 'mnist':
        transforms=[transforms.ToTensor()]
    
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=True, download=True,
                            transform=torchvision.transforms.Compose(transforms),
                            target_transform=ReshapeTransformTarget(10)),
                            batch_size = args.batchSize, shuffle=True)
    
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True,
                            transform=torchvision.transforms.Compose(transforms),
                            target_transform=ReshapeTransformTarget(10)),
                            batch_size = args.test_batchSize, shuffle=True)
    
    elif args.dataset == 'cifar10':
        train_transforms=[transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomCrop(size=[32,32], padding = 4, padding_mode='edge'),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))]
        test_transforms=[transforms.ToTensor(),
                    transforms.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))]
    
    
        train_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                            transform=torchvision.transforms.Compose(train_transforms),
                            target_transform=ReshapeTransformTarget(10)),
                            batch_size = args.batchSize, shuffle=True)
    
        test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                            transform=torchvision.transforms.Compose(test_transforms),
                            target_transform=ReshapeTransformTarget(10)),
                            batch_size = args.test_batchSize, shuffle=True)

if args.activationFun == 'hardsigm':
    def rho(x):
        return x.clamp(min = 0).clamp(max = 1)
    def rhop(x):
        return (x >= 0) & (x <= 1)

elif args.activationFun == 'heaviside':
    def rho(x):
        return (((torch.sign(x)+1)/2)*(x != 0).float() + (x == 0).float()).float()
    def rhop(x):
        return ((x >= -0.5) & (x <= 0.5)).float()

        

if __name__ == '__main__':
    args.layersList.reverse()
        
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    #'''Train a fully-connected architecture - binary and full-precision or binary activation '''
    if args.archi == 'fc':
        if args.binary_settings == "bin_W":
            net = Network_fc_bin_W(args)
        elif args.binary_settings == "bin_W_N":
            net = Network_fc_bin_W_N(args)
            
        if net.cuda:
            net.to(net.device)
            
        BASE_PATH, name = createPath(args)
        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe_fc(BASE_PATH, args, net)
    
        ave_train_error_list, ave_test_error_list = [], []
        single_train_error_list, single_test_error_list = [], []
        train_loss_list, test_loss_list = [], []

        for epoch in tqdm(range(args.epochs)):
            net.epoch = epoch+1
            ave_train_error, single_train_error, train_loss, nb_changes_epoch = train_fc(net, args, train_loader, epoch, optim = args.optim)
            ave_test_error, single_test_error, test_loss = test_fc(net, args, test_loader)
    
            ave_train_error_list.append(ave_train_error.cpu().item())
            ave_test_error_list.append(ave_test_error.cpu().item())
            
            single_train_error_list.append(single_train_error.cpu().item())
            single_test_error_list.append(single_test_error.cpu().item())
    
            train_loss_list.append(train_loss.cpu().item())
            test_loss_list.append(test_loss.cpu().item())
    
            DATAFRAME = updateDataframe_fc(BASE_PATH, args, DATAFRAME, net, ave_train_error_list, ave_test_error_list, single_train_error_list, single_test_error_list, train_loss_list, test_loss_list, nb_changes_epoch)

            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_error': ave_train_error,
                        'test_loss': test_loss.item(),
                        'test_error': ave_test_error
                        },  BASE_PATH + prefix + 'checkpoint.pt')
         
    #'''Train a convolutionnal architecture - binary and full-precision or binary activation '''
    elif args.archi == 'conv':
        if args.binary_settings == "bin_W":
            net = Network_conv_bin_W(args)
        elif args.binary_settings == "bin_W_N":
            net = Network_conv_bin_W_N(args)
            
        if net.cuda:
            net.to(net.device)
            
        BASE_PATH, name = createPath(args)
        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe_conv(BASE_PATH, args, net)
    
        ave_train_error_list, ave_test_error_list = [], []
        single_train_error_list, single_test_error_list = [], []
        train_loss_list, test_loss_list = [], []

        for epoch in tqdm(range(args.epochs)):
            net.epoch = epoch+1
            ave_train_error, single_train_error, train_loss, nb_changes_epoch_fc, nb_changes_epoch_conv = train_conv(net, args, train_loader, epoch, optim = args.optim)
            ave_test_error, single_test_error, test_loss = test_conv(net, args, test_loader, optim = args.optim)
    
            ave_train_error_list.append(ave_train_error.cpu().item())
            ave_test_error_list.append(ave_test_error.cpu().item())
            
            single_train_error_list.append(single_train_error.cpu().item())
            single_test_error_list.append(single_test_error.cpu().item())
    
            train_loss_list.append(train_loss.cpu().item())
            test_loss_list.append(test_loss.cpu().item())
    
            DATAFRAME = updateDataframe_conv(BASE_PATH, args, DATAFRAME, net, ave_train_error_list, ave_test_error_list, single_train_error_list, single_test_error_list, train_loss_list, test_loss_list, nb_changes_epoch_fc, nb_changes_epoch_conv)
            # torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'train_loss': train_loss.item(),
                        'train_error': ave_train_error,
                        'test_loss': test_loss.item(),
                        'test_error': ave_test_error
                        },  BASE_PATH + prefix + 'checkpoint.pt')


# win: cd Desktop/Code/Quantized-Network
# mac: cd Desktop/Thèse/Code/Binarized-EP/Joint-Binarized-Weights-EP