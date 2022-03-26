"""
Implementation of HMMs

We take the following view:
* a  [probabilistic model]  =        a  [class]
* an [inference operation]  =        a  [public member function]
*    [inference operation]  includes    [partition / marginal / argmax / sampling]
* an [inference algorithm]  =        a  [private member function] 

For the HMM model, we have the following correspondance:
  inference operation          inference algorithm
* partition             <--->  forward
* marginal              <--->  forward-backward
* max                   <--->  viterbi
* argmax                <--->  viterbi-backtracking
* sampling              <--->  ancestral sampling
* conditional sampling  <--->  forward-filtering backward-sampling
"""
import numpy as np
from scipy.special import logsumexp


class HMM(object):
    def __init__(self, initial, transition, emission):
        """
        Args:
            initial: size=[num_state]
            transition: size=[num_state, num_state] from state -> to state
            emission: size=[num_observation, num_state]
        """
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_state = transition.shape[0]

        # assume the end state is the last state
        # self.end_state = self.num_state - 1
        return 

    def _forward(self, x):
        """Forward algorithm for computing the alpha and the partition,
        implemented in the log space

        Args:
          x: size=[max_len]

        Returns:
          alpha: size=[max_len, num_state]
          Z: float
        """
        T = len(x)
        N = self.num_state

        alpha = np.zeros((T, N)) 
        alpha[0] = self.emission[x[0]] + self.initial
        for t in range(1, T):
            emission_t = self.emission[x[t]]
            alpha[t] = logsumexp(alpha[t - 1].reshape(N, 1) +
                                 self.transition + 
                                 emission_t.reshape(1, N)
                                 , axis=0)

        Z = logsumexp(alpha[T - 1], axis=0)
        return alpha, Z

    def _backward(self, x):
        """Backward algorithm for computing the beta and the marginals,
        implemented in the log space,
        could be replaced by automatic differentiation

        Args:
          x: size=[max_len]

        Returns:
          beta: size=[max_len, num_state]
        """
        T = len(x)
        N = self.num_state

        beta = np.zeros((T, N))

        # t = [T - 2, T - 3 ,..., 0]
        for t in range(T - 2, -1, -1):
            emission_t_1 = self.emission[x[t + 1]]
            beta[t] = logsumexp(beta[t + 1].reshape(1, N) + 
                                self.transition + 
                                emission_t_1.reshape(1, N), 
                                axis=1)
        return beta

    def _viterbi(self, x):
        """Viterbi algorithm with back-tracking for computing the most probable
        latent state sequence.

        Args:
          x: size=[max_len]

        Returns:
          max_z: size=[max_len]
          max_p: float
        """
        T = len(x)
        N = self.num_state

        max_s = np.zeros((T, N))
        max_ptr = np.zeros((T, N)) # look up table 
        max_s[0] = self.initial + self.emission[x[0]]
        for t in range(1, T):
            emission_t = self.emission[x[t]]
            log_phi_t = self.transition + emission_t.reshape(1, N)
            max_s[t] = np.max(max_s[t-1].reshape(N, 1) + log_phi_t, axis=0)
            max_ptr[t] = np.argmax(max_s[t-1].reshape(N, 1) + log_phi_t, axis=0)
        
        # max_p = np.max(max_s[T - 1] + self.transition[:, self.end_state])
        max_p = np.max(max_s[T - 1])

        # backtracking
        max_z = np.zeros(T).astype(int)
        # max_z[T - 1] = np.argmax(max_s[T - 1] + self.transition[:, self.end_state])
        max_z[T - 1] = np.argmax(max_s[T - 1])
        for t in range(T - 2, -1, -1):
            # print(t + 1)
            # print(max_z[t + 1])
            max_z[t] = max_ptr[t + 1, max_z[t + 1]]
        return max_z, max_p

    def partition(self, x):
        """Log partition the of HMM

        Args:
          x: size=[max_len]

        Returns:
          log_z: float
        """
        _, log_z = self._forward(x)
        return log_z

    def marginal(self, x):
        """Marginal distribution of the latent sequences given the observation x

        Args:
          x: size=[max_len]

        Returns:
          node_marginal, size=[max_len, num_states] # p(h_t | x)
          edge_marginal, size=[max_len - 1, num_states, num_states] p(h_t, h_{t + 1} | x)
        """
        alpha, log_px = self._forward(x)
        beta = self._backward(x)
        node_marginal = alpha + beta - log_px

        T = alpha.shape[0]
        N = alpha.shape[1]
        emission = self.emission[x]  # size = [T, num_state]

        # log edge marginal probability at step t from state i to state j is a T * N * N tensor
        # log p(t, i, j) = alpha(t, i) + transition(i, j) + emission(t + 1, j) + beta(t, j) - Z
        if(T >= 2):
            edge_marginal = alpha[:-1].reshape(T - 1, N, 1) +\
                            self.transition.reshape(1, N, N) +\
                            emission[1:].reshape(T - 1, 1, N) +\
                            beta[1:].reshape(T - 1, 1, N) - log_px
        else: edge_marginal = None
        return alpha, beta, log_px, node_marginal, edge_marginal

    def argmax(self, x):
        """Most probable latent sequence given the observation x
        
        Args:
          x: size=[max_len]

        Returns:
          marginal: size=[batch, max_len, num_state]
        """
        max_z, max_log_prob = self._viterbi(x)
        return max_z, max_log_prob

    def log_prob(self, x):
        """Log probability of a given pair of observed x and latent z
        
        Args:
          x: size=[max_len]

        Returns:
          log_prob: size=[batch]
        """
        _, log_px = self._forward(x)
        return log_px

    def get_delta_param(self, initial, transition, emission):
        assert (initial.shape == self.initial.shape) and \
            (transition.shape == self.transition.shape) and \
            (emission.shape == self.emission.shape), """
            Please make sure the inputs have the same shape to the current\
                parameters.
            """
            
        params_current = np.vstack([self.initial, self.transition, self.emission]).flatten()
        params_new = np.vstack([initial, transition, emission]).flatten()

        return np.linalg.norm(params_new - params_current, ord=2)
