# Training Dynamical Binary Neural Networks with Equilibrium Propagation
This repository contains the code to reproduce the results of the paper "Training Dynamical Binary Neural Network with Equilibrium Propagation".

The project contains the following files:

+ `main.py`: file that centralises the argument of the parser and execute the code. Loads differentially the following files depending on the architecture (fc or conv) and the settings of EP: (binary synapses & full-precision activations) or (binary synapses & binary activations).

+ `Network.py`: contains the network classes.

+ `Tools.py`: contains the functions to run on the networks as well as some useful functions for data analysis.

+ `Plotfunction.py`: contains the functions to plot the results from the results files. 

![GitHub Logo](/schema-section4.png)<!-- .element height="10%" width="10%" -->


## Package requirements

Our code is compatible with Python 3.6 and 3.7.
A suitable environment to run the code can be created in two different ways:

+ with Anaconda:
	```
	conda create --name myenv python=3.6
	conda activate myenv
	conda install -c conda-forge matplotlib
	conda install pytorch torchvision -c pytorch
	```
+ With virtualenv, using the requirements.txt provided in the repo:
	```
	virtualenv myenv
	source myenv/bin/activate
	pip install -r requirements.txt
	```

##  Details about `main.py`

main.py proceeds with the following operations:

+ Parses arguments provided in the command line
	
+ Loads the relevant packages depending on the architecture specified by the command line.
	
+ Loads the dataset depending on the dataset specified by the command line.
	
+ Builds the network from Network.py.
	
+ Trains the network and tests it over epochs and store training data in the dedicated files.
	
	
The parser accepts the following arguments:

+ Hardware settings:

|Arguments|Description|Example|
|-------|------|------|
|`device`| ID (0, 1, ...) of the GPU if provided - -1 for CPU| `--device 0`|

+ Architecture settings:

|Arguments|Description|Example|
|-------|------|------|
|`archi`|Type of architecure we want to train: fc or conv| TBD|
|`layersList`|List of fully connected layers in the network (classifier layer for a conv architecture)| TBD|
|`convList`|List of the conv layers with the numbers of channels specified - the number of channels of the input has to be specified| TBD|
|`padding`|TBD| TBD|
|`kernelSize`|Size of the kernel used for the convolutions| TBD|
|`Fpool`|Size of the pooling kernel| TBD|

+ EqProp settings:

|Arguments|Description|Example|
|-------|------|------|
|`activationFun`|Activation function used in the network| TBD|
|`T`|Number of time steps for the free phase| TBD|
|`Kmax`|Number of time steps for the nudge phase| TBD|
|`beta`|Nudging parameter| TBD|
		
+ Binary optimization settings:	

|Arguments|Description|Example|
|-------|------|------|
|`gamma`|Low pass filter parameter for the fc architecture| TBD|
|`gamma_classi`|Low pass filter parameter for the classifier of the conv architecture| TBD|
|`gamma_conv`|Low pass filter parameter for the conv part of the conv architecture| TBD|
|`gradThreshold`|List of thresholds used in each layer for the fc architecture| TBD|	
|`classi_threshold`|List of thresholds used for the classifier of the conv architecture| TBD|
|`conv_threshold`|List of thresholds used for the conv part of the conv architecture| TBD|
	
	
+ Training settings:

|Arguments|Description|Example|
|-------|------|------|
|`dataset`|Selects which dataset to select to train the network on|TBD|
|`lrBias`|List of the learning for the biases|TBD|
|`batchSize`|Size of training mini-batches|TBD|
|`test_batchSize`|Size of testing mini-batches|TBD|
|`epochs`|Number of epochs to train the network|TBD|

+ Scaling factors learning settings: 


|Arguments|Description|Example|
|-------|------|------|
|`learnAlpha`|Boolean that specify if we learn or let fixed the scaling factors|TBD|
|`lrAlpha`|Learning rates for each scaling factor|TBD|
|`decayLrAlpha`|Quantity by how much we decay the scaling factors|TBD|
|`epochDecayLrAlpha`|Tells when to decay the learning rates for the scaling factors|TBD|	

	
	
## Details about `Network.py`:
Each file contains a class of a network. Each class inherits from a nn.Module.

Each network has the following built-in functions:

+ `init`: specifies the parameter and the architecture of the network.

+ `stepper`: computes the update of the network between two time steps.

+ `forward`: runs the dynamics of the network over t times steps - runs the free and the nudge phase.

+ `computeGradient`: compute the instantaneous gradient for each parameter and accumulate/ filter it.

+ `updateWeights`: update all parameters in the network according to the gradient computed in computeGradient - binary optimization for the weights & SGD for the biases.

+ `initHidden`: initialize the state of network before each free phase.
    
    

## Details about `Tools.py`:
Each file contains the same functions but adapted for each architecture.

+ `train_bin`: trains the network over the whole training dataset once. Run the free and nudge phase and update the weights. Track the number of changes in the weights.

+ `test_bin`: tests the network over the whole testing dataset once.

+ `initDataframe`: inits the dataframe where we store the training data.

+ `updateDataframe`: updates the same dataframe after each epoch: update train error, test error, number of changes of weights for each layer.

+ `createPath`: creates the path and the folder where the training data will be stored. Copy the plotFunction.py file in the same folder. Create a folder specific to the training 'S-X' in this general folder.

+ `saveHyperparameters`: creates a .txt file with all hyperparameters in the parser in the folder 'S-X'.


## Binary Synapses & Full-precision activations: Commands to be run in the terminal to reproduce the results of the paper (Section 3):


* Fully connected architecture:

  + 1 hidden layer with 4096 neurons, fixed scaling factors:

    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W --layersList 784 4096 10 --activationFun hardsigm --T 50 --Kmax 10 --beta 0.3 --random_beta 1 --gamma 1e-5 1e-4 --gradThreshold 5e-7 5e-7 --dataset mnist --lrBias 0.025 0.05 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 0
    ```

  + 1 hidden layer with 4096 neurons, learnt scaling factors:

    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W --layersList 784 100 10 --activationFun hardsigm --T 50 --Kmax 10 --beta 0.3 --random_beta 1 --gamma 1e-5 1e-4 --gradThreshold 5e-7 5e-7 --dataset mnist --lrBias 0.025 0.05 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 1 --lrAlpha 1e-4 1e-4
    ```

  + 2 hidden layers with 4096-4096 neurons, fixed scaling factors:
    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W --layersList 784 4096 4096 10 --activationFun hardsigm --T 250 --Kmax 10 --beta 0.3 --random_beta 1 --gamma 5e-6 2e-5 2e-5 --gradThreshold 5e-7 5e-7 5e-7 --dataset mnist --lrBias 0.025 0.05 0.1 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 0
    ```

  + 2 hidden layers with 4096-4096 neurons, learnt scaling factors:
    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W --layersList 784 4096 4096 10 --activationFun hardsigm --T 250 --Kmax 10 --beta 0.3 --random_beta 1 --gamma 5e-6 2e-5 2e-5 --gradThreshold 5e-7 5e-7 5e-7 --dataset mnist --lrBias 0.025 0.05 0.1 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 1 --lrAlpha 1e-2 1e-2 1e-2
    ```

* Convolutional architecture:

  + 1-32-64-10(fc) architecture with MNIST, fixed scaling factors:

    ```
    python main.py --device 0 --optim ep --archi conv --binary_settings bin_W --layersList 10 --convList 64 32 1 --padding 2 --kernelSize 5 --Fpool 2 --activationFun hardsigm --T 150 --Kmax 10 --beta 0.3 --random_beta 1 --classi_gamma 5e-8 --conv_gamma 5e-8 5e-8 --classi_threshold 1e-8 --conv_threshold 1e-8 1e-8 --dataset mnist --lrBias 0.025 0.05 0.1 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 0
    ```

  + 1-32-64-10(fc) architecture with MNIST, learnt scaling factors:

    ```
    python main.py --device 0 --optim ep --archi conv --binary_settings bin_W --layersList 10 --convList 64 32 1 --padding 2 --kernelSize 5 --Fpool 2 --activationFun hardsigm --T 150 --Kmax 10 --beta 0.3 --random_beta 1 --classi_gamma 5e-8 --conv_gamma 5e-8 5e-8 --classi_threshold 1e-8 --conv_threshold 1e-8 1e-8 --dataset mnist --lrBias 0.025 0.05 0.1 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 1 --lrAlpha 1e-3 1e-3 1e-3
    ```

  + 3-64-128-256-256-10(fc) architecture with CIFAR-10, fixed scaling factors:

    ```
    python main.py --device 0 --optim ep --archi conv --binary_settings bin_W --layersList 10 --convList 256 256 128 64 3 --padding 2 --kernelSize 5 --Fpool 2 --activationFun hardsigm --T 150 --Kmax 10 --beta 0.3 --random_beta 1 --classi_gamma 2e-7 --conv_gamma 2e-7 2e-7 2e-7 2e-7 --classi_threshold 1e-8 --conv_threshold 1e-8 1e-8 1e-8 1e-8 --dataset cifar10 --lrBias 0.025 0.05 0.1 0.2 0.4 --batchSize 64 --test_batchSize 512 --epochs 500  --learnAlpha 0
    ```

  + 3-64-128-256-256-10(fc) architecture with CIFAR-10, learnt scaling factors:

    ```
    python main.py --device 0 --optim ep --archi conv --binary_settings bin_W --layersList 10 --convList 256 256 128 64 3 --padding 2 --kernelSize 5 --Fpool 2 --activationFun hardsigm --T 150 --Kmax 10 --beta 0.3 --random_beta 1 --classi_gamma 2e-7 --conv_gamma 2e-7 2e-7 2e-7 2e-7 --classi_threshold 1e-8 --conv_threshold 1e-8 1e-8 1e-8 1e-8 --dataset cifar10 --lrBias 0.025 0.05 0.1 0.2 0.4 --batchSize 64 --test_batchSize 512 --epochs 500 --learnAlpha 1 --learnAlpha 1 --lrAlpha 1e-4 1e-2 1e-2 1e-2 1e-2 --decayLrAlpha 10 --epochDecayLrAlpha 10
    ```

## Binary Synapses & Binary activations: Commands to be run in the terminal to reproduce the results of the paper (Section 4):


* Fully connected architecture:

  + 1 hidden layer with 8192 neurons, fixed scaling factors:

    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W_N --layersList 784 8192 100 --expand_output 10 --activationFun heaviside --T 20 --Kmax 10 --beta 2 --random_beta 1 --gamma_neur 0.5 --gamma 2e-6 2e-6 --gradThreshold 2.5e-7 2e-7 --dataset mnist --lrBias 1e-7 1e-7 --batchSize 64 --test_batchSize 512 --epochs 100 --learnAlpha 0 --rho_threshold 0.5
    ```

  + 1 hidden layer with 8192 neurons, learnt scaling factors:
    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W_N --layersList 784 100 100 --expand_output 10 --activationFun heaviside --T 20 --Kmax 10 --beta 2 --random_beta 1 --gamma_neur 0.5 --gamma 2e-6 2e-6 --gradThreshold 2.5e-7 2e-7 --dataset mnist --lrBias 1e-7 1e-7 --batchSize 64 --test_batchSize 512 --epochs 100 --learnAlpha 1 --lrAlpha 1e-6 1e-6 --rho_threshold 0.5
    ```

  + 2 hidden layers with 8192-8192 neurons, fixed scaling factors:
    ```
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W_N --layersList 784 8192 8192 8000 --expand_output 800 --activationFun heaviside --T 30 --Kmax 160 --beta 2 --random_beta 0 --gamma 1e-6 1e-6 1e-6 --gradThreshold 2e-8 1e-8 5e-8 --dataset mnist --lrBias 1e-6 1e-6 1e-6 --batchSize 64 --test_batchSize 512 --epochs 100 --learnAlpha 0 --rho_threshold 0.5
    ```

  + 2 hidden layers with 8192-8192 neurons, learnt scaling factors:
    ``` 
    python main.py --device 0 --optim ep --archi fc --binary_settings bin_W_N --layersList 784 8192 8192 8000 --expand_output 800 --activationFun heaviside --T 30 --Kmax 160 --beta 2 --random_beta 0 --gamma 1e-6 1e-6 1e-6 --gradThreshold 2e-8 1e-8 5e-8 --dataset mnist --lrBias 1e-6 1e-6 1e-6 --batchSize 64 --test_batchSize 512 --epochs 100 --learnAlpha 1  --lrAlpha 1e-8 1e-8 1e-8 --rho_threshold 0.5
    ```

* Convolutional architecture:

  + 1-32-64-10(fc) architecture with MNIST, fixed scaling factors:
    ```
    python main.py --device 0 --optim ep --archi conv --binary_settings bin_W_N --layersList 700 -convList 512 256 1  --expand_output 70 --padding 1 --kernelSize 5 --Fpool 3 --activationFun heaviside --T 100 --Kmax 50 --beta 1 --random_beta 1 --classi_gamma 5e-8 --conv_gamma 5e-8 5e-8 --classi_threshold 8e-8 --conv_threshold 8e-8 2e-7 --dataset mnist --lrBias 2e-6 5e-6 1e-5 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 0 --rho_threshold 0.5
    ```
  + 1-32-64-10(fc) architecture with MNIST, learnt scaling factors:
    ```
    python main.py --device 0 --optim ep --archi conv --layersList 700 -convList 512 256 1  --expand_output 70 --padding 1 --kernelSize 5 --Fpool 3 --activationFun heaviside --T 100 --Kmax 50 --beta 1 --random_beta 1 --classi_gamma 5e-8 --conv_gamma 5e-8 5e-8 --classi_threshold 8e-8--conv_threshold 8e-8 2e-7 --dataset mnist --lrBias 2e-6 5e-6 1e-5 --batchSize 64 --test_batchSize 512 --epochs 50 --learnAlpha 1 --lrAlpha 1e-5 1e-3 1e-3 --rho_threshold 0.5
    ```

