# File defining the network and the oscillators composing the network
from scipy import*
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

try:
    from main import rho, rhop1, rhop2
except:
    from main import rho, rhop

class Network(nn.Module):
    ''' Define the network studied
    '''
    def __init__(self, args):

        super(Network, self).__init__()

        self.T = args.T
        self.Kmax = args.Kmax
        self.beta = torch.tensor(args.beta)
        self.gamma_neur = args.gamma_neur
        self.clamped = args.clamped
        self.batchSize = args.batchSize

        if (args.activationFun == 'sign' and args.rho_threshold == 0) or (args.activationFun == 'heavyside' and args.rho_threshold == 0.):
            self.neuronMin = -0.5
            self.neuronMax = 0.5
        elif (args.activationFun == 'sign' and args.rho_threshold != 0) or (args.activationFun == 'heavyside' and args.rho_threshold != 0.):
            self.neuronMin = 0
            self.neuronMax = 1

        self.activationFun = args.activationFun
        self.epoch = 0
        self.model = args.model
        self.weightOffset_tab = []
        #we initialize accGradients to [] that we fill in at the first weights update!
        self.accGradients = []
        self.accGradientsnonFiltered = []
        self.threshold = args.gradThreshold

        self.W = nn.ModuleList(None)
        with torch.no_grad():
            print(args.layersList)
            for i in range(len(args.layersList)-1):
                self.W.extend([nn.Linear(args.layersList[i+1], args.layersList[i], bias = True)])
                # torch.nn.init.zeros_(self.W[-1].bias)

                self.weightOffset = round((torch.norm(self.W[-1].weight, p = 1)/self.W[-1].weight.numel()).item(), 4)
                print("weightOffset = " +str(self.weightOffset))

                #init weights to their alpha value
                self.W[-1].weight.data = torch.sign(self.W[-1].weight.data)*self.weightOffset
                self.weightOffset_tab.append(self.weightOffset)

        if args.device >= 0 and torch.cuda.is_available():
            self.device = torch.device("cuda:"+str(args.device)+")")
            self.cuda = True
        else:
            self.device = torch.device("cpu")
            self.cuda = False


    def getBinState(self, states, phi, args):
        '''
        Return the binary states from pre-activations stored in 'states' and binarization error from 'phi'
        '''
        bin_states = states.copy()

        #we compute binary states from accumulated pre_activation - we use the bias only during this computation
        for layer in range(len(states)-1):
            bin_states[layer] = rho(states[layer] + phi[layer] - args.rho_threshold)

        bin_states[-1] = states[-1]

        return bin_states


    def stepperAveProto(self, args, s, phi, target = None, beta = 0):
        '''
        EP based model - prototypical settings, ie not energy based dynamics!
        with moving average filter for pre-activations
        '''
        pre_act = s.copy()
        #get binary states at t-1:
        bin_states = self.getBinState(s, phi, args)

        #computing pre-activation for every layer weights + bias
        pre_act[0] = self.W[0](bin_states[1])
        pre_act[0] = rhop(s[0] - args.rho_threshold)*pre_act[0]
        if beta != 0:
            pre_act[0] = pre_act[0] + beta*(target*self.neuronMax-s[0])

        for layer in range(1, len(s)-1):
            #previous layer contribution: weights + bias
            pre_act[layer] =  self.W[layer](bin_states[layer+1])
            #next layer contribution
            pre_act[layer] += torch.mm(bin_states[layer-1], self.W[layer-1].weight)
            #multiply with Indicatrice(pre-activation)
            pre_act[layer] = rhop(s[layer] - args.rho_threshold)*pre_act[layer]

        #updating each accumulated pre-activation
        for layer in range(len(s)-1):
            #moving average filter for the pre_activations
            s[layer] =  (1-self.gamma_neur)*s[layer] + self.gamma_neur*pre_act[layer]

            #clamping on pre-activations
            if self.clamped:
                s[layer] = s[layer].clamp(self.neuronMin, self.neuronMax)

        #compute new binary states
        new_bin_states = self.getBinState(s, phi, args)
        for layer in range(len(bin_states)-1):
            print("#changement d'activation layer#"+str(layer)+" = "+str(abs(new_bin_states[layer] - bin_states[layer]).sum()))
            phi[layer] = phi[layer] + s[layer] - new_bin_states[layer]
        del bin_states, new_bin_states

        return s, phi


    def forward(self, args, s, phi, seq = None,  beta = 0, target = None, tracking = False):
        '''
        Relaxation function
        '''
        T, Kmax = self.T, self.Kmax
        h, y = [], []

        with torch.no_grad():
            if beta == 0:
                # Free phase
                for t in range(T):
                    print("========= T = " +str(t) + "==========")
                    s, phi = self.stepperAveProto(args, s, phi)

                    if tracking:
                        y.append(s[0][0][5].item())
                        h.append(s[1][0][5].item())
            else:
                # Nudged phase
                for t in range(Kmax):
                    print("========= K = " +str(t) + "==========")
                    s, phi = self.stepperAveProto(args, s, phi, target = target, beta = self.beta)

                    if tracking:
                        y.append(s[0][0][5].item())
                        h.append(s[1][0][5].item())
        if tracking:
            return s, phi, y, h
        else:
            return s, phi


    def computeGradients(self, args, s, phi, seq, phi_seq, method = None):
        '''
        Compute EQ gradient to update the synaptic weight - from Binary neurons
        '''
        batch_size = s[0].size(0)
        coef = 1/(self.beta*batch_size).float()
        gradW, gradBias = [], []

        if args.model == 'ave_prototypical':
            bin_states_0, bin_states_1 = self.getBinState(seq, phi_seq, args), self.getBinState(s, phi, args)

        for layer in range(len(s)-1):
            gradW.append( coef * (torch.mm(torch.transpose(bin_states_1[layer], 0, 1), bin_states_1[layer+1]) - torch.mm(torch.transpose(bin_states_0[layer], 0, 1), bin_states_0[layer+1])))
            gradBias.append( coef * (bin_states_1[layer] -  bin_states_0[layer]).sum(0))


        if self.accGradients == []:
            self.accGradients = [args.gamma[i]*w_list for (i, w_list) in enumerate(gradW)]
            self.accGradientsnonFiltered = [w_list for w_list in gradW]

        else:
            self.accGradients = [torch.add((1-args.gamma[i])*x1, args.gamma[i]*x2) for i, (x1, x2) in enumerate(zip(self.accGradients, gradW))]
            self.accGradientsnonFiltered = [w_list for w_list in gradW]

        gradW = self.accGradients

        return gradW, gradBias

    def updateWeight(self, epoch, s, phi, seq, phi_seq, args):
        '''
        Update weights with Hebbian learning rule
        '''
        with torch.no_grad():
            gradW, gradBias = self.computeGradients(args, s, phi, seq, phi_seq)

            nb_changes = []

            for i in range(len(s)-1):
                threshold_tensor = self.threshold[i]*torch.ones(self.W[i].weight.size()).to(self.device)

                modify_weights = -1*torch.sign((-1*torch.sign(self.W[i].weight)*gradW[i] > threshold_tensor).float()-0.5)

                self.W[i].weight.data = self.W[i].weight.data * modify_weights.float()

                nb_changes.append(torch.sum(abs(modify_weights-1).float()).item()/2)

                #update bias
                self.W[i].bias.data += args.lrBias[i] * gradBias[i]

        return nb_changes


    def initHidden(self, args, data, testing = False):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        state, phi = [], []
        size = data.size(0)

        for layer in range(len(args.layersList)-1):
            state.append(self.neuronMax*torch.ones(size, args.layersList[layer], requires_grad = False))
            phi.append(torch.zeros(size, args.layersList[layer], requires_grad = False))

        if args.model == 'ave_prototypical':
            if args.activationFun == 'sign':
                state.append(2*(torch.round(data)-0.5).float())

            else:
                state.append(torch.round(data).float())

        else:
            state.append(data.float())

        return state, phi




