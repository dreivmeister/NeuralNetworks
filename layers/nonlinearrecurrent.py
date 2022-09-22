import numpy as np
import itertools
from layers.nonlinearrecurrent_helper import TensorLinear, RecurrentStateUnfold, LogisticClassifier



class RNN(object):
    def __init__(self, nb_of_inputs, nb_of_outputs, nb_of_states, nb_of_rec_lay, sequence_len, tensor_order):
        self.sequence_len = sequence_len

        #Tensor Input layer
        self.inputLayer = TensorLinear(nb_of_inputs, nb_of_states, tensor_order)
        #List of recurrent layers (RecurrentStateUnfold)
        self.recurrentLayers = [RecurrentStateUnfold(nb_of_states, sequence_len)]*int(nb_of_rec_lay)
        #Tensor Output layer
        self.outputLayer = TensorLinear(nb_of_states, nb_of_outputs, tensor_order)
        #Classification
        self.classifier = LogisticClassifier()
        
    def forward(self, X):
        #Linear input transformation
        recIn = self.inputLayer.forward(X)
        #Propagate through recurrent layers
        S = self.recurrentLayers[0].forward(recIn)

        if len(self.recurrentLayers) > 1:
            for recLay in self.recurrentLayers[1:]:
                S = recLay.forward(S)
        #Linear output transformation
        Z = self.outputLayer.forward(S[:,1:self.sequence_len+1,:])
        Y = self.classifier.forward(Z)

        return recIn, S, Z, Y
    
    def backward(self, X, Y, recIn, S, T):
        #Classifier
        gZ = self.classifier.backward(Y, T)
        #Output
        gRecOut, gWout, gBout = self.outputLayer.backward(
            S[:,1:self.sequence_len+1,:], gZ)

        #Recurrent Layers
        gRecs = []
        gRnnIn, gWrec, gBrec, gS0 = self.recurrentLayers[0].backward(recIn, S, gRecOut)
        gRecs.append([gRnnIn, gWrec, gBrec, gS0])
        if len(self.recurrentLayers) > 1:
            for recLay in reversed(self.recurrentLayers[1:]):
                gRnnIn, gWrec, gBrec, gS0 = recLay.backward(recIn, S, gRecOut)
                gRecs.append([gRnnIn, gWrec, gBrec, gS0])

        #Input
        gX, gWin, gBin = self.inputLayer.backward(X, gRecs[-1][0])
    
        return gWout, gBout, gWrec, gBrec, gWin, gBin, gS0, gRecs

    def getOutput(self, X):
        """Get the output probabilities of input X."""
        recIn, S, Z, Y = self.forward(X)
        return Y
    
    def getBinaryOutput(self, X):
        """Get the binary output of input X."""
        return np.around(self.getOutput(X))

    def getParamGrads(self, X, T):
        """Return the gradients with respect to input X and 
        target T as a list. The list has the same order as the 
        get_params_iter iterator."""
        recIn, S, Z, Y = self.forward(X)
        gWout, gBout, gWrec, gBrec, gWin, gBin, gS0, gRecs = self.backward(
            X, Y, recIn, S, T)

        ndi = []

        for lay in gRecs:
            ndi.append(np.nditer(lay[3]))

        ndi.append(np.nditer(gBin))
        ndi.append(np.nditer(gWin))
        
        for lay in gRecs:
            ndi.append(np.nditer(lay[1]))
            ndi.append(np.nditer(lay[2]))
        
        ndi.append(np.nditer(gWout))
        ndi.append(np.nditer(gBout))

        return [g for g in itertools.chain.from_iterable(ndi)]

    def loss(self, Y, T):
        """Return the loss of input X w.r.t. targets T."""
        return self.classifier.loss(Y, T)
    

    def get_params_iter(self):
        """Return an iterator over the parameters.
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        ndi = []
        
        for recLay in self.recurrentLayers:
            ndi.append(np.nditer(recLay.S0, op_flags=['readwrite']))
        ndi.append(np.nditer(self.inputLayer.W, op_flags=['readwrite']))
        ndi.append(np.nditer(self.inputLayer.b, op_flags=['readwrite']))
        for recLay in self.recurrentLayers:
            ndi.append(np.nditer(recLay.W, op_flags=['readwrite']))
            ndi.append(np.nditer(recLay.b, op_flags=['readwrite']))
        ndi.append(np.nditer(self.outputLayer.W, op_flags=['readwrite']))
        ndi.append(np.nditer(self.outputLayer.b, op_flags=['readwrite']))

        return itertools.chain.from_iterable(ndi)





# Create dataset
nb_train = 2000  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 7  # Length of the binary sequence

def create_dataset(nb_samples, sequence_len):
    """Create a dataset for binary addition and 
    return as input, targets."""
    max_int = 2**(sequence_len-1) # Maximum integer that can be added
     # Transform integer in binary format
    format_str = '{:0' + str(sequence_len) + 'b}'
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    # Input samples
    X = np.zeros((nb_samples, sequence_len, nb_inputs))
    # Target samples
    T = np.zeros((nb_samples, sequence_len, nb_outputs))
    # Fill up the input and target matrix
    for i in range(nb_samples):
        # Generate random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        # Fill current input and target row.
        # Note that binary numbers are added from right to left, 
        #  but our RNN reads from left to right, so reverse the sequence.
        X[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1)]))
        X[i,:,1] = list(
            reversed([int(b) for b in format_str.format(nb2)]))
        T[i,:,0] = list(
            reversed([int(b) for b in format_str.format(nb1+nb2)]))
    return X, T

    # Show an example input and target
def printSample(x1, x2, t, y=None):
    """Print a sample in a more visual way."""
    x1 = ''.join([str(int(d)) for d in x1])
    x1_r = int(''.join(reversed(x1)), 2)
    x2 = ''.join([str(int(d)) for d in x2])
    x2_r = int(''.join(reversed(x2)), 2)
    t = ''.join([str(int(d[0])) for d in t])
    t_r = int(''.join(reversed(t)), 2)
    if not y is None:
        y = ''.join([str(int(d[0])) for d in y])
    print(f'x1:   {x1:s}   {x1_r:2d}')
    print(f'x2: + {x2:s}   {x2_r:2d}')
    print(f'      -------   --')
    print(f't:  = {t:s}   {t_r:2d}')
    if not y is None:
        y_r = int(''.join(reversed(y)), 2)
        print(f'y:  = {y:s}   {y_r:2d}')

# Create training samples
X_train, T_train = create_dataset(nb_train, sequence_len)
print(f'X_train tensor shape: {X_train.shape}')
print(f'T_train tensor shape: {T_train.shape}')




# # Set hyper-parameters
# lmbd = 0.5  # Rmsprop lambda
# learning_rate = 0.05  # Learning rate
# momentum_term = 0.80  # Momentum term
# eps = 1e-6  # Numerical stability term to prevent division by zero
# mb_size = 100  # Size of the minibatches (number of samples)
# nb_of_states = 3 # Number of states in the recurrent layer
# nb_of_rec_lay = 1


def objective_function(x):
    # Create the network
    lmbd = x[0]
    learning_rate = x[1]  # Learning rate
    momentum_term = x[2]  # Momentum term
    mb_size = int(x[3])  # Size of the minibatches (number of samples)
    nb_of_states = int(x[4]) # Number of states in the recurrent layer
    nb_of_rec_lay = int(x[5])
    R = RNN(nb_of_inputs=2, nb_of_outputs=1, nb_of_states=nb_of_states, nb_of_rec_lay=nb_of_rec_lay, sequence_len=7, tensor_order=3)
    # Set the initial parameters
    # Number of parameters in the network
    nbParameters =  sum(1 for _ in R.get_params_iter())
    # Rmsprop moving average
    maSquare = [0.0 for _ in range(nbParameters)]
    Vs = [0.0 for _ in range(nbParameters)]  # Momentum


    #Training
    # Create a list of minibatch losses to be plotted
    ls_of_loss = [R.loss(R.getOutput(X_train[0:100,:,:]), T_train[0:100,:,:])]
    # Iterate over some iterations
    nb_of_iter = 100
    for i in range(nb_of_iter):
        # Iterate over all the minibatches
        for mb in range(nb_train // mb_size):
            X_mb = X_train[mb:mb+mb_size,:,:]  # Input minibatch
            T_mb = T_train[mb:mb+mb_size,:,:]  # Target minibatch
            V_tmp = [v * momentum_term for v in Vs]
            # Update each parameters according to previous gradient
            for pIdx, P in enumerate(R.get_params_iter()):
                P += V_tmp[pIdx]
            # Get gradients after following old velocity
            # Get the parameter gradients
            backprop_grads = R.getParamGrads(X_mb, T_mb)    
            # Update each parameter seperately
            #print(len(backprop_grads), len(maSquare))
            for pIdx, P in enumerate(R.get_params_iter()):
                # Update the Rmsprop moving averages
                maSquare[pIdx] = lmbd * maSquare[pIdx] + (
                    1-lmbd) * backprop_grads[pIdx]**2
                # Calculate the Rmsprop normalised gradient
                pGradNorm = ((
                    learning_rate * backprop_grads[pIdx]) / np.sqrt(
                    maSquare[pIdx]) + 1e-6)
                # Update the momentum
                Vs[pIdx] = V_tmp[pIdx] - pGradNorm     
                P -= pGradNorm   # Update the parameter
            # Add loss to list to plot
            ls_of_loss.append(R.loss(R.getOutput(X_mb), T_mb))


    #Problem: list from getParamGrads is shorter than list from get_params_iter
    # Create test samples
    nb_test = 50
    Xtest, Ttest = create_dataset(nb_test, sequence_len)
    # Push test data through network
    Y = R.getBinaryOutput(Xtest)
    Yf = R.getOutput(Xtest)


    # Print out all test examples
    count_correct = 0
    for i in range(Xtest.shape[0]):
        if (Ttest[i,:,:] == Y[i,:,:]).any():
            count_correct += 1
    if count_correct == 0:
        return 0
    return (nb_test/count_correct)*100



def random_num(start, stop, step):
    return np.random.choice(np.arange(start, stop, step))


#lmbd, learning_rate, momentum_term, mb_size, nb_of_states, nb_of_rec_lay
#[ 0.7   0.04  0.6  10.    4.    2.  ]
#[[start, stop, step],[start,stop,step]]
def generate_random_samples(num_samples, num_params, params):
    X = np.zeros((num_samples, num_params))

    for i in range(num_samples):
        for j in range(num_params):
            X[i,j] = random_num(params[j][0],params[j][1],params[j][2])
    
    return X

params = [[0.1, 1.0, 0.1],[0.01, 0.1, 0.01],[0.1,1.0,0.1],[10, 100, 10],[1,10,1],[1,4,1]]

