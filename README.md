# Binarized Equilibirum Propagation for facilitated on-chip learning - Binary weights EqProp

Here are listed description of the code uploaded and the commands to be run to reproduce the results shown in the paper - for the binary-weights section only.

## I - Package requirements

Our code is compatible with Python 3.6 and 3.7.

The package requirements are listed in the file 'requirements.txt'.

## II - Files

#### - The repo contains the following files:

	i) main.py: file that centralise the argument of the parser and execute the code. Loads differentially the following files depending on the architecture (fc or conv).
	
	ii) Network_fc.py: file that builds the class of a fully connected network and all its functions.
	
	iii) Tools_fc.py: file that train the fc-network, test it and save all data & parameters in a dedicated folder.
	
	iv) Network_conv.py: file that builds the class of a convolutionnal network and all its functions.
	
	v) Tools_conv.py: file that train the conv-network, test it and save all data & parameters in a dedicated folder.
	
	vi) plotFunctions.py: file used to plot the training curves as they appear in the paper. It is copied in each new folder with new parameters.


## III - Run main.py file

#### - main.py does the following operations:

	i) Parses arguments provided in the command line
	
	ii) Loads the relevant packages depending on the architecture specified by the command line.
	
	iii) Loads the dataset depending on the dataset specified by the command line.
	
	iv) Builds the network with Network_fc.py or Network_conv.py.
	
	v) Trains the network and tests it over epochs and store training data in the dedicated files.
	
	
 #### - The parser accepts the following arguments:
	i) Hardware settings:
	
	--device: ID (0, 1, ...) of the GPU if provided - -1 for CPU
	
	ii) Architecture settings:
	
	--archi: type of architecure we want to train: fc or conv
	
	--layersList: list of fully connected layers in the network	(classifier layer for a conv architecture)
	
	--convList: list of the conv layers with the numbers of channels specified - the number of channels of the input has to be specified
	
	--padding: 
	
	--kernelSize: size of the kernel used for the convolutions
	
	--Fpool: size of the pooling kernel
	
	iii) EqProp settings:
	
	--activationFun: activation function used in the network
	
	--T: number of time steps for the free phase
	
	--Kmax: number of time stpes for the nudge phase
	
	--beta: nudging parameter
	
    iv) Binary optimization settings:
	
	--gamma: low pass filter parameter for the fc architecture
	
	--gamma_classi: low pass filter parameter for the classifier of the conv architecture
	
	--gamma_conv: low pass filter parameter for the conv part of the conv architecture
	
	--gradThreshold: list of thresholds used in each layer for the fc architecture
	
	--classi_threshold: list of thresholds used for the classifier of the conv architecture
	
	--conv_threshold: list of thresholds used for the conv part of the conv architecture
	
	v) Training settings:
	
	--dataset: which dataset to select to train the network on
	
	--lrBias: list of the learning for the biases
	
	--batchSize: size of training mini-batches
	
	--test_batchSize: size of testing mini-batches
	
	--epochs: number of epochs to train the network

	
	
## IV - Details about Network_x.py file:
Each file contains a class of a network. Each class inherits from a nn.Module.

Each network has the following built-in functions:

i) init: specifies the parameter and the architecture of the network

ii) stepper: computes the update of the network between two time steps

iii) forward: runs the dynamics of the network over t times steps - run the free and the nudge phase

iv) computeGradient: compute the instantaneous gradient for each parameter and accumulate/ filter it

v) updateWeights: update all parameters in the network according to the gradient computed in computeGradient - binary optimization for the weights & SGD for the biases

vi) initHidden: init the state of network before each free phase to 1
    
    

## V - Details about Tools_x.py file:
Each file contains the same functions but adapted for each architecture.

i) train_bin: train the network over the whole training dataset once. Run the free and nudge phase and update the weights. Track the number of changes in the weights.

ii) test_bin: test the network over the whole testing dataset once.

iii) initDataframe: init the dataframe where we store the training data

iv) updateDataframe: update the same dataframe after each epoch: update train error, test error, number of changes of weights for each layer

v) createPath: create the path and the folder where the training data will be stored. Copy the plotFunction.py file in the same folder. Create a folder specific to the training 'S-X' in this general folder.

vi) saveHyperparameters: create a .txt file with all hyperparameters in the parser in the folder 'S-X'.


## VI - Command to be run in the terminal to reproduce the results of the paper: