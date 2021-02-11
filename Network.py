from scipy import*
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from main import rho, rhop

#==============================================================================================================  
#====================== FC Architecture - Binary Synapses - Full-precision activation =========================   
#==============================================================================================================  
 
 
class Network_fc_bin_W(nn.Module):
    
    def __init__(self, args):

        super(Network_fc_bin_W, self).__init__()
        
        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device)+")")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False

        self.T = args.T
        self.Kmax = args.Kmax
        
        self.beta = torch.tensor(args.beta)
        self.batchSize = args.batchSize

        self.neuronMin = 0
        self.neuronMax = 1
        
        self.activationFun = args.activationFun
        self.epoch = 0
        
        self.threshold = args.gradThreshold
        self.weightOffset_tab = []
        
        #we initialize accGradients to [] and we fill it in at the first weights update and then use it to store the momentum
        self.accGradients = []
        self.accGradientsnonFiltered = []

        #Module that contains all weight and bias parameters
        self.W = nn.ModuleList(None)
        with torch.no_grad():
            for i in range(len(args.layersList)-1):
                self.W.extend([nn.Linear(args.layersList[i+1], args.layersList[i], bias = True)])

                #Init the scaling factor with the XNOR-Net method
                self.weightOffset = round((torch.norm(self.W[-1].weight, p = 1)/self.W[-1].weight.numel()).item(), 4)
                self.weightOffset_tab.append(self.weightOffset)
                self.W[-1].weight.data = torch.sign(self.W[-1].weight.data)*self.weightOffset

    def stepper(self, args, s, target = None, beta = 0):
        '''
        Prototypical dynamics
        '''
        dsdt = []
        dsdt.append(rho(self.W[0](s[1])))

        if beta != 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, len(s)-1):
            dsdt.append( rho(self.W[i](s[i+1]) + torch.mm(s[i-1], self.W[i-1].weight) ))

        for layer in range(len(s)-1):
            s[layer] = dsdt[layer]
            s[layer] = s[layer].clamp(self.neuronMin, self.neuronMax)

        return s


    def forward(self, args, s, seq = None,  beta = 0, target = None, optim = 'ep'):
        '''
        Relaxation function
        '''
        T, Kmax = self.T, self.Kmax

        if (optim == 'ep'):
            with torch.no_grad():
                if beta == 0:
                    # Free phase
                    for t in range(T):
                            s = self.stepper(args, s)
                else:
                    # Nudged phase
                    for t in range(Kmax):
                            s = self.stepper(args, s, target = target, beta = beta)
                        
        elif (optim == 'bptt'):
            for t in range(T):
                if t == T - Kmax - 1:
                    for i in range(len(s)):
                        s[i] = s[i].detach()
                        s[i].requires_grad = True

                s = self.stepper(args, s)

            return s

        return s

    def computeGradientsBPTT(self, s, args):
        gradW, gradBias = [], []
        
        for params in self.W:
            gradW.append(-params.weight.grad)
            gradBias.append(-params.bias.grad)

        if self.accGradients == []:
            self.accGradients = [args.gamma[i]*w_list for (i, w_list) in enumerate(gradW)]
            self.accGradientsnonFiltered = [w_list for w_list in gradW]

        else:
            self.accGradients = [torch.add((1-args.gamma[i])*x1, args.gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.accGradients, gradW))]
            self.accGradientsnonFiltered = [w_list for w_list in gradW]

        gradW = self.accGradients

        return gradW, gradBias
        

    def computeGradients(self, args, s, seq):
        batch_size = s[0].size(0)
        coef = 1/(self.beta*batch_size)
        gradW, gradBias, gradAlpha = [], [], []

        with torch.no_grad():
            for layer in range(len(s)-1):
                gradW.append( coef * (torch.mm(torch.transpose(s[layer], 0, 1), s[layer+1]) - torch.mm(torch.transpose(seq[layer], 0, 1), seq[layer+1])))
                gradBias.append( coef * (s[layer] -  seq[layer]).sum(0))
                if args.learnAlpha == 1:
                    gradAlpha.append(0.5 * coef * (torch.diag(torch.mm(s[layer], torch.mm(self.W[layer].weight, s[layer+1].T))).sum() - torch.diag(torch.mm(seq[layer], torch.mm(self.W[layer].weight, seq[layer+1].T))).sum()))
        
            if self.accGradients == []:
                self.accGradients = [args.gamma[i]*w_list for (i, w_list) in enumerate(gradW)]
                self.accGradientsnonFiltered = [w_list for w_list in gradW]
    
            else:
                self.accGradients = [torch.add((1-args.gamma[i])*x1, args.gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.accGradients, gradW))]
                self.accGradientsnonFiltered = [torch.add(x1, x2) for i, (x1, x2) in enumerate(zip(self.accGradientsnonFiltered, gradW))]
    
    
            gradW = self.accGradients

        return gradW, gradBias, gradAlpha


    def updateWeight(self, epoch, s, seq, args, optim = 'ep'):
        '''
        Update parameters: weights, biases, scaling factors if relevant
        '''
        if (optim == 'ep'):
            gradW, gradBias, gradAlpha = self.computeGradients(args, s, seq)
        elif (optim == 'bptt'):
            gradW, gradBias = self.computeGradientsBPTT(s, args)

        nb_changes = []
        
        with torch.no_grad():
            for i in range(len(s)-1):
                #update weights
                threshold_tensor = self.threshold[i]*torch.ones(self.W[i].weight.size(0), self.W[i].weight.size(1)).to(self.device)
                modify_weights = -1*torch.sign((-1*torch.sign(self.W[i].weight)*gradW[i] > threshold_tensor).int()-0.5)
                self.W[i].weight.data = torch.mul(self.W[i].weight.data, modify_weights)
                nb_changes.append(torch.sum(abs(modify_weights-1)).item()/2)
                
                #update biases
                self.W[i].bias.data += args.lrBias[i] * gradBias[i]
                
                #update scaling factors
                if args.learnAlpha == 1:
                    print(self.weightOffset_tab[i])
                    self.weightOffset_tab[i] += args.lrAlpha[i] * gradAlpha[i]
                    print(self.weightOffset_tab[i])
                    self.W[i].weight.data = self.weightOffset_tab[i] * torch.sign(self.W[i].weight)

        return nb_changes


    def initHidden(self, args, data, testing = False):
        '''
        Initialize the state of the network + input layer
        '''
        state = []
        size = data.size(0)

        for layer in range(len(args.layersList)):
            state.append(torch.zeros(size, args.layersList[layer], requires_grad = False))

        state[len(state)-1] = data.float()

        return state
   
#==============================================================================================================  
#=========================== FC Architecture - Binary Synapses - Binary activation ============================   
#==============================================================================================================  
    
        
class Network_fc_bin_W_N(nn.Module):
    def __init__(self, args):

        super(Network_fc_bin_W_N, self).__init__()

            
        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device))
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False
            
        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.gamma_neur = args.gamma_neur
        self.batchSize = args.batchSize

        self.neuronMin = 0
        self.neuronMax = 1

        self.activationFun = args.activationFun
        self.epoch = 0
        #we initialize accGradients to [] that we fill in at the first weights update!
        self.accGradients = []
        self.accGradientsnonFiltered = []
        self.threshold = args.gradThreshold
        self.weightOffset_tab = []

        self.W = nn.ModuleList(None)
        with torch.no_grad():
            for i in range(len(args.layersList)-1):
                self.W.extend([nn.Linear(args.layersList[i+1], args.layersList[i], bias = True)])

                weightOffset = round((torch.norm(self.W[-1].weight, p = 1)/self.W[-1].weight.numel()).item(), 4)
                self.weightOffset_tab.append(weightOffset)
                self.W[-1].weight.data = torch.sign(self.W[-1].weight.data)*weightOffset


    def getBinState(self, states, args):
        bin_states = states.copy()
 
        for layer in range(len(states)-1):
            bin_states[layer] = rho(states[layer] - args.rho_threshold)

        #we copy full-precision input data
        bin_states[-1] = states[-1]

        return bin_states


    def stepper(self, args, s, seq = None, target = None, beta = 0):
        pre_act = s.copy()
        bin_states = self.getBinState(s, args)

        #computing pre-activation for every layer weights + bias
        pre_act[0] = self.W[0](bin_states[1])
        pre_act[0] = rhop(s[0] - args.rho_threshold)*pre_act[0]
        if beta != 0:
            pre_act[0] = pre_act[0] + beta*(target*self.neuronMax-s[0])

        for layer in range(1, len(s)-1):
            #previous layer contribution: weights + bias
            pre_act[layer] =  self.W[layer](bin_states[layer+1])
            # next layer contribution
            pre_act[layer] += torch.mm(bin_states[layer-1], self.W[layer-1].weight)
            #multiply with Indicatrice(pre-activation)
            pre_act[layer] = rhop(s[layer] - args.rho_threshold)*pre_act[layer]

        #updating each accumulated pre-activation
        for layer in range(len(s)-1):
            #moving average filter for the pre_activations
            s[layer] =  (1-self.gamma_neur)*s[layer] + self.gamma_neur*pre_act[layer]
            #clamping on pre-activations
            s[layer] = s[layer].clamp(self.neuronMin, self.neuronMax)

        return s


    def forward(self, args, s, seq = None,  beta = 0, target = None, optim = 'ep'):
        '''
        No support for BPTT yet
        '''
        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            if beta == 0:
                for t in range(T):
                    s = self.stepper(args, s)

            else:
                for t in range(Kmax):
                    s = self.stepper(args,s, target = target, beta = beta, seq = seq)

        return s


    def computeGradients(self, args, s, seq):
        batch_size = s[0].size(0)
        coef = 1/(self.beta*batch_size).float()
        gradW, gradBias, gradAlpha = [], [], []

        with torch.no_grad():
            bin_states_0, bin_states_1 = self.getBinState(seq, args), self.getBinState(s, args)

            for layer in range(len(s)-1):
                gradW.append( coef * (torch.mm(torch.transpose(bin_states_1[layer], 0, 1), bin_states_1[layer+1]) - torch.mm(torch.transpose(bin_states_0[layer], 0, 1), bin_states_0[layer+1])))
                gradBias.append( coef * (bin_states_1[layer] -  bin_states_0[layer]).sum(0))
                gradAlpha.append(0.5 * coef * (torch.diag(torch.mm(bin_states_1[layer], torch.mm(self.W[layer].weight, bin_states_1[layer+1].T))).sum() - torch.diag(torch.mm(bin_states_0[layer], torch.mm(self.W[layer].weight, bin_states_0[layer+1].T))).sum()))

            if self.accGradients == []:
                self.accGradients = [args.gamma[i]*w_list for (i, w_list) in enumerate(gradW)]
                self.accGradientsnonFiltered = [w_list for w_list in gradW]

            else:
                self.accGradients = [torch.add((1-args.gamma[i])*x1, args.gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.accGradients, gradW))]
                self.accGradientsnonFiltered = [w_list for w_list in gradW]

        gradW = self.accGradients

        return gradW, gradBias, gradAlpha
        

    def updateWeight(self, epoch, s, seq, args, optim = 'ep'):

        gradW, gradBias, gradAlpha = self.computeGradients(args, s, seq)

        nb_changes = []
        with torch.no_grad():
            for i in range(len(s)-1):
                #update binary weights
                threshold_tensor = self.threshold[i]*torch.ones(self.W[i].weight.size()).to(self.device)

                modify_weights = -1*torch.sign((-1*torch.sign(self.W[i].weight)*gradW[i] > threshold_tensor).float()-0.5)

                self.W[i].weight.data = self.W[i].weight.data * modify_weights.float()

                nb_changes.append(torch.sum(abs(modify_weights-1).float()).item()/2)

                #update bias
                self.W[i].bias.data += args.lrBias[i] * gradBias[i]

                #update scaling factor
                if args.learnAlpha == 1:
                    self.weightOffset_tab[i] += args.lrAlpha[i] * gradAlpha[i]
                    self.W[i].weight.data = self.weightOffset_tab[i] * torch.sign(self.W[i].weight)

        return nb_changes


    def initHidden(self, args, data, testing = False):
        state = []
        size = data.size(0)

        for layer in range(len(args.layersList)-1):
            state.append(torch.ones(size, args.layersList[layer], requires_grad = False))

        state.append(data.float())

        return state


#==============================================================================================================  
#===================== Conv Architecture - Binary Synapses - Full-precision activation ========================   
#==============================================================================================================  
  
  
class Network_conv_bin_W(nn.Module):

    def __init__(self, args):
        super(Network_conv_bin_W, self).__init__()
        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device))
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False

        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.batchSize = args.batchSize
        
        self.neuronMin = 0
        self.neuronMax = 1
        
        self.activationFun = args.activationFun
        self.epoch = 0

        self.class_accGradients = []
        self.conv_accGradients = []
        
        self.fc_threshold = args.classi_threshold
        self.conv_threshold = args.conv_threshold

        self.kernelSize = args.kernelSize
        self.Fpool = args.Fpool
        self.convList = args.convList
        self.layersList = args.layersList
        
        self.n_cp = len(args.convList) - 1
        self.n_classifier = len(args.layersList)
        
        self.P, P = args.padding, args.padding 

        self.conv = nn.ModuleList([])
        self.fc = nn.ModuleList([])  

        if args.dataset == 'mnist':
            input_size = 28
        elif args.dataset == 'cifar10':
            input_size = 32
        
        self.size_convpool_tab = [input_size] 
        self.size_conv_tab = [input_size]

        with torch.no_grad():
            #Define conv operations
            for i in range(self.n_cp):
                self.conv.append(nn.Conv2d(args.convList[i + 1], args.convList[i], args.kernelSize, padding = P))
                #we binarize weights and scale them channel-wise
                for k in range(args.convList[i]):
                    weightOffset = round((torch.norm(self.conv[-1].weight[k], p = 1)/self.conv[-1].weight[k].numel()).item(), 4)
                    self.conv[-1].weight[k] = weightOffset * torch.sign(self.conv[-1].weight[k])
                            
                self.size_conv_tab.append(self.size_convpool_tab[i] - args.kernelSize + 1 + 2*P)
    
                self.size_convpool_tab.append(int(np.floor((self.size_convpool_tab[i] - args.kernelSize + 1 + 2*P -args.Fpool)/2 + 1)))
            
            self.pool = nn.MaxPool2d(args.Fpool, stride = args.Fpool, return_indices = True)	        
            self.unpool = nn.MaxUnpool2d(args.Fpool, stride = args.Fpool)    
            
            self.size_convpool_tab = list(reversed(self.size_convpool_tab))
            self.size_conv_tab = list(reversed(self.size_conv_tab))
            
            self.nconv = len(self.size_convpool_tab) - 1
            self.layersList.append(args.convList[0]*self.size_convpool_tab[0]**2)
            
            self.nc = len(self.layersList) - 1
        
            for i in range(self.n_classifier):
                self.fc.append(nn.Linear(self.layersList[i + 1], self.layersList[i]))    
                
                weightOffset = round((torch.norm(self.fc[-1].weight, p = 1)/self.fc[-1].weight.numel()).item(), 4)
                self.fc[-1].weight.data = weightOffset * torch.sign(self.fc[-1].weight)


    def stepper(self, args, s, data, inds, target = None, beta = 0): 
        
        dsdt = []
        
        #CLASSIFIER PART
        #last classifier layer
        dsdt.append(-s[0] + rho(self.fc[0](s[1].view(s[1].size(0), -1))))

        if beta != 0:
            dsdt[0] = dsdt[0] + beta*(target - s[0])
             
        #middle classifier layer
        for i in range(1, len(self.layersList) - 1):
            dsdt.append(-s[i] + rho(self.fc[i](s[i + 1].view(s[i + 1].size(0), -1)) + torch.mm(s[i - 1], self.fc[i - 1].weight)))

        #CONVOLUTIONAL PART
        #last conv layer
        s_pool, ind = self.pool(self.conv[0](s[self.nc + 1]))
        inds[self.nc] = ind
        dsdt.append(-s[self.nc] + rho(s_pool + torch.mm(s[self.nc - 1], self.fc[-1].weight).view(s[self.nc].size())))

        del s_pool, ind
        
        #middle layers
        for i in range(1, self.nconv - 1):	
            s_pool, ind = self.pool(self.conv[i](s[self.nc + i + 1]))
            inds[self.nc + i] = ind
            
            if inds[self.nc + i - 1] is not None:      

                output_size = [s[self.nc + i - 1].size(0), s[self.nc + i - 1].size(1), self.size_conv_tab[i - 1], self.size_conv_tab[i - 1]]
                                                            
                s_unpool = F.conv_transpose2d(self.unpool(s[self.nc + i - 1], inds[self.nc + i - 1], output_size = output_size), 
                                                weight = self.conv[i - 1].weight, padding = self.P)                                                
                                                                                                                                                               
            dsdt.append(-s[self.nc + i] + rho(s_pool + s_unpool))
            del s_pool, s_unpool, ind, output_size
            
        #first conv layer
        s_pool, ind = self.pool(self.conv[-1](data))
        inds[-1] = ind
        if inds[-2] is not None:
            output_size = [s[-2].size(0), s[-2].size(1), self.size_conv_tab[-3], self.size_conv_tab[-3]]
            s_unpool = F.conv_transpose2d(self.unpool(s[-2], inds[-2], output_size = output_size), weight = self.conv[-2].weight, padding = self.P)
        dsdt.append(-s[-1] + rho(s_pool + s_unpool))
        del s_pool, s_unpool, ind, output_size
        
        for i in range(len(s)):
            s[i] = s[i] + dsdt[i]   #on a += car dans dsdt on fait -s[i]!         

        del dsdt  
        return s, inds



    def forward(self, args, s, data, inds,  beta = 0, target = None, optim = 'ep'):
        T, Kmax = self.T, self.Kmax
        
        if (optim == 'ep'):
            with torch.no_grad():
                if beta == 0:
                    # Free phase
                    for t in range(T):
                        s, inds = self.stepper(args, s, data, inds)
    
                else:
                    # Nudged phase
                    for t in range(Kmax):
                        s, inds = self.stepper(args, s, data, inds, target = target, beta = beta)
                        
        elif (optim == 'bptt'):
            with torch.no_grad():
                for t in range(T-Kmax):
                    s, inds = self.stepper(args, s, data, inds)

            for i in range(len(s)):
                s[i] = s[i].detach()
                s[i].requires_grad = True
                
            for t in range(Kmax):
                s, inds = self.stepper(args, s, data, inds)

        return s, inds


    def computeBPTTGradients(self, args):
        
        gradfc, gradfc_bias = [], []
        gradconv, gradconv_bias = [], []
        
        for params in self.fc:
            gradfc.append(-params.weight.grad)
            gradfc_bias.append(-params.bias.grad)
            
        for params in self.conv:
            gradconv.append(-params.weight.grad)
            gradconv_bias.append(-params.bias.grad)
            
        with torch.no_grad():
            if (self.class_accGradients == []) or (self.conv_accGradients == []):
                self.class_accGradients = [args.classi_gamma[i]*w_list for (i, w_list) in enumerate(gradfc)]
                self.conv_accGradients  = [args.conv_gamma[i]*w_list for (i, w_list) in enumerate(gradconv)]
    
            else:
                self.class_accGradients = [torch.add((1-args.classi_gamma[i])*x1, args.classi_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.class_accGradients, gradfc))]
                self.conv_accGradients  = [torch.add((1-args.conv_gamma[i])*x1, args.conv_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.conv_accGradients, gradconv))]
    
            gradfc   = self.class_accGradients
            gradconv = self.conv_accGradients

        return gradfc, gradfc_bias, gradconv, gradconv_bias


    def computeGradients(self, args, s, seq, inds, indseq, data):

        batch_size = s[0].size(0)
        coef = 1/(self.beta*batch_size)
        
        gradfc, gradfc_bias = [], []
        gradconv, gradconv_bias = [], []
        gradAlpha_fc, gradAlpha_conv = [], []
               
        #CLASSIFIER       
        for i in range(self.nc-1):

            gradfc.append(coef*(torch.mm(torch.transpose(s[i].view(seq[i].size(0),-1), 0, 1), s[i + 1].view(seq[i].size(0),-1)) - torch.mm(torch.transpose(seq[i].view(seq[i].size(0),-1), 0, 1), seq[i + 1].view(seq[i].size(0),-1))))          
            gradfc_bias.append(coef*(s[i] - seq[i]).sum(0))   
            gradAlpha_fc.append(coef*(torch.diag(torch.mm(s[i], torch.mm(self.W[layer].weight, s[i+1].T))).sum() - torch.diag(torch.mm(seq[i], torch.mm(self.W[layer].weight, seq[i+1].T))).sum()))                                                                        
        
        gradfc.append(coef*(torch.mm(torch.transpose(s[self.nc - 1], 0, 1), s[self.nc].view(s[self.nc].size(0), -1)) - torch.mm(torch.transpose(seq[self.nc - 1], 0, 1), seq[self.nc].view(seq[self.nc].size(0), -1))))   
               
        gradfc_bias.append(coef*(s[self.nc - 1] - seq[self.nc - 1]).sum(0))
        
        gradAlpha_fc.append(coef*(torch.diag(torch.mm(s[self.nc-1], torch.mm(self.fc[self.nc-1].weight, s[self.nc].view(s[self.nc].size()[0], -1).T))).sum() - torch.diag(torch.mm(seq[self.nc-1], torch.mm(self.fc[self.nc-1].weight, seq[self.nc].view(seq[self.nc].size()[0], -1).T))).sum()))                                                                        
        
                                                                             
        #CONVOLUTIONAL
        for i in range(self.nconv-1):

            output_size = [s[self.nc + i].size(0), s[self.nc + i].size(1), self.size_conv_tab[i], self.size_conv_tab[i]]                                            

            gradconv.append(coef*(F.conv2d(s[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                            - F.conv2d(seq[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
            gradconv_bias.append(coef*(self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size) - self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[self.nc + i].size(1), -1).sum(1))
            
            gradAlpha_conv.append(coef*((s[self.nc + i].permute(1,0,2,3)[:,:]*self.pool(self.conv[i](s[self.nc + i + 1]))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)-(seq[self.nc + i].permute(1,0,2,3)[:,:]*self.pool(self.conv[i](seq[self.nc + i + 1]))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)))
            
          
        output_size = [s[-1].size(0), s[-1].size(1), self.size_conv_tab[-2], self.size_conv_tab[-2]]

        gradconv.append(coef*(F.conv2d(data.permute(1, 0, 2, 3), self.unpool(s[-1], inds[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                        - F.conv2d(data.permute(1, 0, 2, 3), self.unpool(seq[-1], indseq[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
        gradconv_bias.append(coef*(self.unpool(s[-1], inds[-1], output_size = output_size) - self.unpool(seq[-1], indseq[-1], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[-1].size(1), -1).sum(1))

        gradAlpha_conv.append(coef*((s[-1].permute(1,0,2,3)[:,:]*self.pool(self.conv[-1](data))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)-(seq[-1].permute(1,0,2,3)[:,:]*self.pool(self.conv[-1](data))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)))

        
        if (self.class_accGradients == []) or (self.conv_accGradients == []):
            self.class_accGradients = [args.classi_gamma[i]*w_list for (i, w_list) in enumerate(gradfc)]
            self.conv_accGradients  = [args.conv_gamma[i]*w_list for (i, w_list) in enumerate(gradconv)]

        else:
            self.class_accGradients = [torch.add((1-args.classi_gamma[i])*x1, args.classi_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.class_accGradients, gradfc))]
            self.conv_accGradients  = [torch.add((1-args.conv_gamma[i])*x1, args.conv_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.conv_accGradients, gradconv))]

        gradfc   = self.class_accGradients
        gradconv = self.conv_accGradients

        
        return gradfc, gradfc_bias, gradconv, gradconv_bias, gradAlpha_fc, gradAlpha_conv


    def updateWeight(self, s, seq, inds, indseq, args, data, optim):
        '''
        Update weights with Hebbian learning rule
        '''
        if optim == 'ep':
            gradfc, gradfc_bias, gradconv, gradconv_bias, gradAlpha_fc, gradAlpha_conv = self.computeGradients(args, s, seq, inds, indseq, data)
        elif optim == 'bptt':
            gradfc, gradfc_bias, gradconv, gradconv_bias = self.computeBPTTGradients(args)
            
        nb_changes_fc = []
        nb_changes_conv = []
        
        with torch.no_grad():
            for i in range(self.nc):   
                threshold_tensor = self.fc_threshold[i]*torch.ones(self.fc[i].weight.size()).to(self.device)
                modify_weights = -1*torch.sign((-1*torch.sign(self.fc[i].weight)*gradfc[i] >= threshold_tensor).int()-0.5)
                
                self.fc[i].weight.data = torch.mul(self.fc[i].weight.data, modify_weights)
    
                nb_changes_fc.append(torch.sum(abs(modify_weights-1)).item()/2)
                
                self.fc[i].bias += args.lrBias[i]*gradfc_bias[i]
                
                #rescale weights
                if args.learnAlpha == 1:
                    scaling = self.fc[0].weight[0,0].cpu().abs()
                    self.fc[i].weight.data = torch.sign(self.fc[i].weight) * (scaling + args.lrAlpha[i]*gradAlpha_fc[i])

            for i in range(self.nconv):
                threshold_tensor = self.conv_threshold[i]*torch.ones(self.conv[i].weight.size()).to(self.device)
                modify_weights = -1*torch.sign((-1*torch.sign(self.conv[i].weight)*gradconv[i] >= threshold_tensor).int()-0.5)
                
                self.conv[i].weight.data = torch.mul(self.conv[i].weight.data, modify_weights)
    
                nb_changes_conv.append(torch.sum(abs(modify_weights-1)).item()/2)
                
                self.conv[i].bias += args.lrBias[i + len(self.fc)]*gradconv_bias[i] 
                
                #rescale the weights
                if args.learnAlpha == 1:
                    scaling = self.conv[i].weight[:,0,0,0].abs()
                    size = self.conv[i].weight.size()
                    expected_size = size[1]*size[2]*size[3]
                    self.conv[i].weight.data = self.conv[i].weight.sign() * (scaling + args.lrAlpha[i+len(self.fc)]*gradAlpha_conv[i]).repeat_interleave(expected_size).view(self.conv[i].weight.size())

        return nb_changes_fc, nb_changes_conv


    def initHidden(self, args, data):

        s, inds = [], []
        batch_size = data.size(0)

        for layer in range(self.nc):
            s.append(0.5*torch.ones(batch_size, args.layersList[layer], requires_grad = True))
            inds.append(None)
            
        for layer in range(self.nconv):
            s.append(0.5*torch.ones(batch_size, self.convList[layer], self.size_convpool_tab[layer], self.size_convpool_tab[layer], requires_grad = True))
            inds.append(None)

        return s, inds
        

#==============================================================================================================  
#===================== Conv Architecture - Binary Synapses - Binary activation ========================   
#==============================================================================================================  

class Network_conv_bin_W_N(nn.Module):

    def __init__(self, args):

        super(Network_conv_bin_W_N, self).__init__()
        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device))
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False

        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.batchSize = args.batchSize
        
        self.neuronMin = 0
        self.neuronMax = 1
            
        self.activationFun = args.activationFun
        self.epoch = 0

        self.class_accGradients = []
        self.conv_accGradients = []
        self.fc_threshold = args.classi_threshold
        self.conv_threshold = args.conv_threshold

        self.kernelSize = args.kernelSize
        self.Fpool = args.Fpool
        self.convList = args.convList
        self.layersList = args.layersList
        
        self.n_cp = len(args.convList) - 1
        self.n_classifier = len(args.layersList)
        
        P = args.padding

        self.P = args.padding

        self.conv = nn.ModuleList([])
        self.fc = nn.ModuleList([])  

        if args.dataset == "mnist":
            input_size = 28
        elif args.dataset == "cifar10":
            input_size = 32
        
        self.size_convpool_tab = [input_size] 
        self.size_conv_tab = [input_size]

        for i in range(self.n_cp):
            self.conv.append(nn.Conv2d(args.convList[i + 1], args.convList[i], args.kernelSize, padding = P))
            for k in range(args.convList[i]):
                weightOffset = round((torch.norm(self.conv[-1].weight[k], p = 1)/self.conv[-1].weight[k].numel()).item(), 4)
                self.conv[-1].weight[k] = weightOffset * torch.sign(self.conv[-1].weight[k])
                
            # torch.nn.init.zeros_(self.conv[-1].bias)
                        
            self.size_conv_tab.append(int((self.size_convpool_tab[i] + 2*P - args.kernelSize)/1 + 1 ))
            
            self.size_convpool_tab.append(int((self.size_conv_tab[-1]-args.Fpool+2*0)/args.Fpool)+1)
        
        self.pool = nn.MaxPool2d(args.Fpool, stride = args.Fpool, return_indices = True)	        
        self.unpool = nn.MaxUnpool2d(args.Fpool, stride = args.Fpool)    

        self.size_convpool_tab.reverse()
        self.size_conv_tab.reverse()
        
        
        self.nconv = len(self.size_convpool_tab) - 1

        self.layersList.append(args.convList[0]*self.size_convpool_tab[0]**2)

        self.nc = len(self.layersList) - 1
        
        for i in range(self.n_classifier):
            self.fc.append(nn.Linear(self.layersList[i + 1], self.layersList[i]))    
            weightOffset = round((torch.norm(self.fc[-1].weight, p = 1)/self.fc[-1].weight.numel()).item(), 4)

            self.fc[-1].weight.data = weightOffset * torch.sign(self.fc[-1].weight)
            torch.nn.init.zeros_(self.fc[-1].bias)


    def getBinState(self, states, args):
        '''
        Return the binary states from pre-activations stored in 'states'
        '''
        bin_states = states.copy()

        for layer in range(len(states)):
            bin_states[layer] = rho(states[layer] - args.rho_threshold)

        return bin_states


    def stepper(self, args, data, s, inds, target = None, beta = 0):
        pre_act = s.copy()
        bin_states = self.getBinState(s, args)
        
        #CLASSIFIER PART
        #last classifier layer
        pre_act[0] = self.fc[0](bin_states[1].view(bin_states[1].size(0), -1))
        pre_act[0] = rhop(s[0] - args.rho_threshold)*pre_act[0]
        
        if beta != 0:
            pre_act[0] = pre_act[0] + beta*(target*self.neuronMax-s[0])
        
        #middle classifier layer
        for i in range(1, len(self.layersList) - 1):
            pre_act[layer] =  self.fc[layer](bin_states[layer+1].view(bin_states[layer+1].size(0), -1))
            pre_act[layer] += torch.mm(bin_states[layer-1], self.fc[layer-1].weight)
            pre_act[layer] = rhop(s[layer] - args.rho_threshold)*pre_act[layer]

        #CONVOLUTIONAL PART
        #last conv layer
        s_pool, ind = self.pool(self.conv[0](bin_states[self.nc + 1]))
        inds[self.nc] = ind
        pre_act[self.nc] = s_pool 
        pre_act[self.nc] += torch.mm(bin_states[self.nc - 1], self.fc[-1].weight).view(s[self.nc].size())
        pre_act[self.nc] = rhop(s[self.nc] - args.rho_threshold)*pre_act[self.nc]

        del s_pool, ind
        
        #middle layers
        for i in range(1, self.nconv - 1):	
            s_pool, ind = self.pool(self.conv[i](bin_states[self.nc + i + 1]))
            inds[self.nc + i] = ind
            
            if inds[self.nc + i - 1] is not None:      

                output_size = [s[self.nc + i - 1].size(0), s[self.nc + i - 1].size(1), self.size_conv_tab[i - 1], self.size_conv_tab[i - 1]]
                                                            
                s_unpool = F.conv_transpose2d(self.unpool(bin_states[self.nc + i - 1], inds[self.nc + i - 1], output_size = output_size), 
                                                weight = self.conv[i - 1].weight, padding = self.P)                                                
                                                                                                                                                               
            pre_act[self.nc+i] = s_pool 
            pre_act[self.nc+i] += s_unpool
            pre_act[self.nc+i] = rhop(s[self.nc+i] - args.rho_threshold)*pre_act[self.nc+i]

            del s_pool, s_unpool, ind, output_size
            
        #first conv layer
        s_pool, ind = self.pool(self.conv[-1](data))
        inds[-1] = ind
        if inds[-2] is not None:
            output_size = [s[-2].size(0), s[-2].size(1), self.size_conv_tab[-3], self.size_conv_tab[-3]]
            s_unpool = F.conv_transpose2d(self.unpool(bin_states[-2], inds[-2], output_size = output_size), weight = self.conv[-2].weight, padding = self.P)
        pre_act[-1] = s_pool 
        pre_act[-1] += s_unpool
        pre_act[-1] = rhop(s[-1] - args.rho_threshold)*pre_act[-1]
        del s_pool, s_unpool, ind, output_size
        
        
        for layer in range(len(s)):
            s[layer] = (1-args.gamma_neur)*s[layer] + args.gamma_neur*pre_act[layer]   
            s[layer] = s[layer].clamp(self.neuronMin, self.neuronMax)
        
        del pre_act, bin_states
        
        return s, inds


    def forward(self, args, s, data, inds, seq = None, target = None, beta = 0, optim = 'ep'):

        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            if beta == 0:
                for t in range(T):
                    s, inds = self.stepper(args, data, s, inds)

            else:
                for t in range(Kmax):
                    s, inds = self.stepper(args, data, s, inds, target = target, beta = beta)

        return s, inds


    def computeGradients(self, args, s, seq, inds, indseq, data):
        
        batch_size = s[0].size(0)
        coef = 1./float((self.beta*batch_size))
        s = self.getBinState(s, args)
        seq = self.getBinState(seq, args)
        
        gradfc, gradfc_bias = [], []
        gradconv, gradconv_bias = [], []
        gradAlpha_fc, gradAlpha_conv = [], []
               
        #CLASSIFIER       
        for i in range(self.nc-1):
            gradfc.append(coef*(torch.mm(torch.transpose(s[i].view(seq[i].size(0),-1), 0, 1), s[i + 1].view(seq[i].size(0),-1)) - torch.mm(torch.transpose(seq[i].view(seq[i].size(0),-1), 0, 1), seq[i + 1].view(seq[i].size(0),-1))))          
            gradfc_bias.append(coef*(s[i] - seq[i]).sum(0)) 
            gradAlpha_fc.append(coef*(torch.diag(torch.mm(s[i], torch.mm(self.W[layer].weight, s[i+1].T))).sum() - torch.diag(torch.mm(seq[i], torch.mm(self.W[layer].weight, seq[i+1].T))).sum()))                                                                        
        
        gradfc.append(coef*(torch.mm(torch.transpose(s[self.nc - 1], 0, 1), s[self.nc].view(s[self.nc].size(0), -1)) - torch.mm(torch.transpose(seq[self.nc - 1], 0, 1), seq[self.nc].view(seq[self.nc].size(0), -1))))   
               
        gradfc_bias.append(coef*(s[self.nc - 1] - seq[self.nc - 1]).sum(0))
        gradAlpha_fc.append(coef*(torch.diag(torch.mm(s[self.nc-1], torch.mm(self.fc[self.nc-1].weight, s[self.nc].view(s[self.nc].size()[0], -1).T))).sum() - torch.diag(torch.mm(seq[self.nc-1], torch.mm(self.fc[self.nc-1].weight, seq[self.nc].view(seq[self.nc].size()[0], -1).T))).sum())) 
        
                                                                             
        #CONVOLUTIONAL
        for i in range(self.nconv-1):
            output_size = [s[self.nc + i].size(0), s[self.nc + i].size(1), self.size_conv_tab[i], self.size_conv_tab[i]]                                            

            gradconv.append(coef*(F.conv2d(s[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                            - F.conv2d(seq[self.nc + i + 1].permute(1, 0, 2, 3), self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
            gradconv_bias.append(coef*(self.unpool(s[self.nc + i], inds[self.nc + i], output_size = output_size) - self.unpool(seq[self.nc + i], indseq[self.nc + i], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[self.nc + i].size(1), -1).sum(1))
            gradAlpha_conv.append(coef*((s[self.nc + i].permute(1,0,2,3)[:,:]*self.pool(self.conv[i](s[self.nc + i + 1]))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)-(seq[self.nc + i].permute(1,0,2,3)[:,:]*self.pool(self.conv[i](seq[self.nc + i + 1]))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)))
            
          
        output_size = [s[-1].size(0), s[-1].size(1), self.size_conv_tab[-2], self.size_conv_tab[-2]]

        gradconv.append(coef*(F.conv2d(data.permute(1, 0, 2, 3), self.unpool(s[-1], inds[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)
                        - F.conv2d(data.permute(1, 0, 2, 3), self.unpool(seq[-1], indseq[-1], output_size = output_size).permute(1, 0, 2, 3), padding = self.P)).permute(1, 0, 2, 3))
        gradconv_bias.append(coef*(self.unpool(s[-1], inds[-1], output_size = output_size) - self.unpool(seq[-1], indseq[-1], output_size = output_size)).permute(1, 0, 2, 3).contiguous().view(s[-1].size(1), -1).sum(1))
        
        gradAlpha_conv.append(coef*((s[-1].permute(1,0,2,3)[:,:]*self.pool(self.conv[-1](data))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)-(seq[-1].permute(1,0,2,3)[:,:]*self.pool(self.conv[-1](data))[0].permute(1,0,2,3)[:,:]).sum(3).sum(2).sum(1)))

        if (self.class_accGradients == []) or (self.conv_accGradients == []):
            self.class_accGradients = [(1-args.classi_gamma[i])*2*args.classi_gamma[i]*torch.randn(w_list.size()).to(self.device) + args.classi_gamma[i]*w_list for (i, w_list) in enumerate(gradfc)]
            self.conv_accGradients  = [(1-args.conv_gamma[i])*2*args.conv_gamma[i]*torch.randn(w_list.size()).to(self.device) +args.conv_gamma[i]*w_list for (i, w_list) in enumerate(gradconv)]

        else:
            self.class_accGradients = [torch.add((1-args.classi_gamma[i])*x1, args.classi_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.class_accGradients, gradfc))]
            self.conv_accGradients  = [torch.add((1-args.conv_gamma[i])*x1, args.conv_gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.conv_accGradients, gradconv))]

        gradfc   = self.class_accGradients
        gradconv = self.conv_accGradients

        
        return gradfc, gradfc_bias, gradconv, gradconv_bias, gradAlpha_fc, gradAlpha_conv


    def updateWeight(self, s, seq, inds, indseq, args, data, optim = 'ep'):
        
        with torch.no_grad():
            gradfc, gradfc_bias, gradconv, gradconv_bias, gradAlpha_fc, gradAlpha_conv = self.computeGradients(args, s, seq, inds, indseq, data)
            nb_changes_fc, nb_changes_conv  = [], []
        
            for i in range(self.nc): 
                #update weights   
                threshold_tensor = self.fc_threshold[i]*torch.ones(self.fc[i].weight.size()).to(self.device)
                modify_weights = -1*torch.sign((-1*torch.sign(self.fc[i].weight)*gradfc[i] >= threshold_tensor).int()-0.5)
                
                self.fc[i].weight.data = torch.mul(self.fc[i].weight.data, modify_weights)
    
                nb_changes_fc.append(torch.sum(abs(modify_weights-1)).item()/2)
                
                #update bias
                self.fc[i].bias += args.lrBias[i]*gradfc_bias[i]
                
                #rescale weights
                if args.learnAlpha == 1:
                    scaling = self.fc[0].weight[0,0].cpu().abs()
                    self.fc[i].weight.data = torch.sign(self.fc[i].weight) * (scaling + args.lrAlpha[i]*gradAlpha_fc[i])

            for i in range(self.nconv):
                #update weights
                threshold_tensor = self.conv_threshold[i]*torch.ones(self.conv[i].weight.size()).to(self.device)
                modify_weights = -1*torch.sign((-1*torch.sign(self.conv[i].weight)*gradconv[i] >= threshold_tensor).int()-0.5)
                
                self.conv[i].weight.data = torch.mul(self.conv[i].weight.data, modify_weights)
    
                nb_changes_conv.append(torch.sum(abs(modify_weights-1)).item()/2)
                
                #update bias
                self.conv[i].bias += args.lrBias[i + len(self.fc)]*gradconv_bias[i]
                
                #rescale the weights
                if args.learnAlpha == 1:
                    scaling = self.conv[i].weight[:,0,0,0].abs()
                    size = self.conv[i].weight.size()
                    expected_size = size[1]*size[2]*size[3]
                    self.conv[i].weight.data = self.conv[i].weight.sign() * (scaling + args.lrAlpha[i + len(self.fc)]*gradAlpha_conv[i]).repeat_interleave(expected_size).view(self.conv[i].weight.size())

        return nb_changes_fc, nb_changes_conv


    def initHidden(self, args, data, testing = False):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        s, inds = [], []
        size = data.size(0)
       
        for layer in range(self.nc):
            s.append(torch.ones(size, args.layersList[layer], requires_grad = False))
            inds.append(None)
            
        for layer in range(self.nconv):
            s.append(torch.ones(size, self.convList[layer], self.size_convpool_tab[layer], self.size_convpool_tab[layer], requires_grad = False))
            inds.append(None)

        return s, inds









        
        
