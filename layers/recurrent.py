import numpy as np  # Matrix and vector computation package

#Backwardstepping
#Output Gradient
def mse_prime(y, t):
    """
    Gradient of the MSE loss function with respect to the output y.
    """
    return 2. * (y - t)



class LinearRecurrentCell:
    def __init__(self, X, t, W, W_del, W_sgn, eta_p, eta_n) -> None:
        self.X = X
        self.t = t
        self.W = W
        self.W_del = W_del
        self.W_sgn = W_sgn
        self.eta_p = eta_p
        self.eta_n = eta_n



    #Forwardstepping
    def update_state(self, xk, sk, wx, wRec):
        """
        Compute state k from the previous state (sk) and current 
        input (xk), by use of the input weights (wx) and recursive 
        weights (wRec).
        """
        return xk * wx + sk * wRec


    def forward_states(self, X, wx, wRec):
        """
        Unfold the network and compute all state activations 
        given the input X, input weights (wx), and recursive weights 
        (wRec). Return the state activations in a matrix, the last 
        column S[:,-1] contains the final activations.
        """
        # Initialise the matrix that holds all states for all 
        #  input sequences. The initial state s0 is set to 0.
        S = np.zeros((X.shape[0], X.shape[1]+1))
        # Use the recurrence relation defined by update_state to update 
        #  the states trough time.
        for k in range(0, X.shape[1]):
            # S[k] = S[k-1] * wRec + X[k] * wx
            S[:,k+1] = self.update_state(X[:,k], S[:,k], wx, wRec)
        return S




    def backward_gradient(self, X, S, grad_out, wRec):
        """
        Backpropagate the gradient computed at the output (grad_out) 
        through the network. Accumulate the parameter gradients for 
        wX and wRec by for each layer by addition. Return the parameter 
        gradients as a tuple, and the gradients at the output of each layer.
        """
        # Initialise the array that stores the gradients of the loss with 
        #  respect to the states.
        grad_over_time = np.zeros((X.shape[0], X.shape[1]+1))
        grad_over_time[:,-1] = grad_out
        # Set the gradient accumulations to 0
        wx_grad = 0
        wRec_grad = 0
        for k in range(X.shape[1], 0, -1):
            # Compute the parameter gradients and accumulate the results.
            wx_grad += np.sum(
                np.mean(grad_over_time[:,k] * X[:,k-1], axis=0))
            wRec_grad += np.sum(
                np.mean(grad_over_time[:,k] * S[:,k-1]), axis=0)
            # Compute the gradient at the output of the previous layer
            grad_over_time[:,k-1] = grad_over_time[:,k] * wRec
        return (wx_grad, wRec_grad), grad_over_time

    # Define Rprop optimisation function
    def update_rprop(self, X, t, W, W_prev_sign, W_delta, eta_p, eta_n):
        """
        Update RProp values in one iteration.
        Args:
            X: input data.
            t: targets.
            W: Current weight parameters.
            W_prev_sign: Previous sign of the W gradient.
            W_delta: RProp update values (Delta).
            eta_p, eta_n: RProp hyperparameters.
        Returns:
            (W_delta, W_sign): Weight update and sign of last weight
                            gradient.
        """
        # Perform forward and backward pass to get the gradients
        S = self.forward_states(X, W[0], W[1])
        grad_out = mse_prime(S[:,-1], t)
        W_grads, _ = self.backward_gradient(X, S, grad_out, W[1])
        W_sign = np.sign(W_grads)  # Sign of new gradient
        # Update the Delta (update value) for each weight 
        #  parameter seperately
        for i, _ in enumerate(W):
            if W_sign[i] == W_prev_sign[i]:
                W_delta[i] *= eta_p
            else:
                W_delta[i] *= eta_n
        return W_delta, W_sign


    def train(self):
        for i in range(500):
            # Get the update values and sign of the last gradient
            W_delta, W_sign = self.update_rprop(
                self.X, self.t, self.W, self.W_sgn, self.W_del, self.eta_p, self.eta_n)

            self.W_del = W_delta
            self.W_sgn = W_sign

            # Update each weight parameter seperately
            for i, _ in enumerate(self.W):
                self.W[i] -= self.W_sgn[i] * self.W_del[i]
            #ls_of_ws.append((W[0], W[1]))  # Add weights to list to plot

        print(f'Final weights are: wx = {self.W[0]:.4f},  wRec = {self.W[1]:.4f}')
    
    def test(self, test_inpt):
        test_outpt = self.forward_states(test_inpt, self.W[0], self.W[1])[:,-1]
        sum_test_inpt = test_inpt.sum()
        print((
            f'Target output: {sum_test_inpt:d} vs Model output: '
            f'{test_outpt[0]:.2f}'))






#Dataset creation
nb_of_samples = 20
sequence_len = 16
# Create the sequences
X = np.zeros((nb_of_samples, sequence_len))
for row_idx in range(nb_of_samples):
    X[row_idx,:] = np.around(np.random.rand(sequence_len)).astype(int)
# Create the targets for each sequence
y = np.sum(X, axis=1)


#Training
# Perform Rprop optimisation
# Set hyperparameters
eta_p = 1.2
eta_n = 0.5

# Set initial parameters
Weights = [-1.5, 2]  # [wx, wRec]
W_delta = [0.001, 0.001]  # Update values (Delta) for W
W_sign = [0, 0,]  # Previous sign of W

#ls_of_ws = [(W[0], W[1])]  # List of weights to plot


Lr = LinearRecurrentCell(X, y, Weights, W_delta, W_sign, eta_p, eta_n)
Lr.train()


#Testing
test_inpt = np.asmatrix([[0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]])
Lr.test(test_inpt)
