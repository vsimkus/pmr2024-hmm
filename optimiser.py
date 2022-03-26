"""
Learning algorithms for HMM model

Author: Shangmin Guo
Contact: s.guo@ed.ac.uk
"""
import math
from typing import List, Tuple
import numpy as np
from numpy.typing import ArrayLike
from hmm import HMM


class HMMOptimiser(object):
    """Optimiser for the parameters of HMM.

    This class implements the algorithms to optimise the parameters of HMM 
    model in discrete case, with either supervised or unsupervised data. 

    Attributes:
        model: 
            an HMM model from hmm.py
        supervised: 
            a boolean variable that indicates the HMM model is trained in
            supervised or unsupervised fashion
        num_hiddens:
            an integer which is the number of possible hidden states
        num_observations:
            an integer which is the number of possible observed states
    
    Methods:
        __init__:
            initialise the class
        fit:
            fit the parameters of an HMM model on a given data set
        baum_welch:
            fit on unsupervised data
        _initial_params:
            initialise the parameters of HMM model in the Baum-Welch algorithm
        _e_step:
            estimation step of the Baum-Welch algorithm
        _m_step:
            maximisation step of the Baum-Welch algorithm
        _stop_criterion:
            criterion for stopping the iteration in the Baum-Welch algorithm
        counts:
            fit on supervised data
        get_trained_model:
            get an HMM class instance with parameters fit on a given data set
    """
    
    def __init__(self, 
                supervised:bool=False,
                num_hiddens:int=None,
                num_observations:int=None
                ) -> None:
        super().__init__()
        assert num_hiddens is not None and num_observations is not None
        self.model = None
        self.supervised = supervised
        self.num_hiddens = num_hiddens
        self.num_observations = num_observations
        
    def fit(self, data_loader:List) -> None:
        """Fit the parameters on the given data loader.
        
        This method will fit the three parameters of HMM model on the given 
        sequences, i.e. *initial* (distribution), *transition* probability 
        matrix (between hidden states), and *emission* probability matrix (from
        hidden states to observed states).
        
        There are two different types of learning:
        1) supervised, where true hidden states are also provided in the data
            loader;
        2) unsupervised, where only sequences of observed states are provided in
            the data loader.
            
        Note that this method doesn't return the optimised model, instead users
        need to get it through the method "get_trained_model".
            
        Args:
            data_loader:
                a list whose element should be 
                    1) a tuple containing two lists of integers, if learning is
                        supervised;
                    2) a list of integers, if learning is unsupervised
        
        Return:
            None
        """
        
        if self.supervised:
            assert type(data_loader[0]) == tuple and len(data_loader[0]) == 2, \
                """Samples in the data list should contain observations and 
                hiddens at the same time if we want to train the model in a
                supervised way."""
                
            self.counts(data_loader)
        else:
            self.baum_welch(data_loader)
    
    def _e_step(self, data_loader:complex) -> Tuple[float, List, List]:
        """E-step of the Baum-Welch algorithm
        
        This method implements the E-step which estimates the probability 
        distribution over the hidden states given a sequence of observations, 
        i.e. the following two quantities:
            1. $p(h_i, h_{i-1} | D_j; \theta_{old})$;
            2. $p(h_i | D_j; \theta_{old})$.
        Since both of the above values have been calculated in the `marginal` 
        method of the HMM model in `hmm.py`, we can directly get the results
        from it.
        
        Args:
            data_loader:
                a list of lists whose elements are integers.
        
        Return:
            loglikelihood:
                float whose value equals to the average loglikelihood of the 
                input data.
            hk_list:
                a list of np.arrays correspondding to the value of 
                $p(h_i | D_j; \theta_{old})$
            hkk_list:
                a list of np.arrays correspondding to the value of 
                $p(h_i, h_{i-1} | D_j; \theta_{old})$

        """
        
        hk_list = []
        hkk_list = []
        log_ps = []
        
        for o_seq in data_loader:
            # TODO: your code here (keep the left hand side)
            _, _, log_p, hk, hkk = self.model.marginal(o_seq) 
            hk_list.append(hk)
            hkk_list.append(hkk)
            log_ps.append(log_p)
            
        return np.mean(log_ps), hk_list, hkk_list
    
    def _m_step(self, 
                data_loader:object, 
                hk_list:List, 
                hkk_list:List
                ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """ M-step of the Baum-Welch algorithm
        
        This method implements the M-step which maximises the parameters of the
        HMM model (self.model) given observations and the following estimations:
            1. $p(h_i, h_{i-1} | D_j; \theta_{old})$;
            2. $p(h_i | D_j; \theta_{old})$.
        The equations used for updating the parameters are given as following:
            1. $a_k = \frac{1}{n}\sum_{j=1}^{n}p(h_1=k|\mathcal{D}_j;\theta_{\text{old}})$;
            2. $A_{k,k'} = \frac{\sum_{j=1}^{n}\sum_{i=2}^{d}p(h_i=k,h_{i-1}=k'|\mathcal{D}_j;\theta_{\text{old}})}{\sum_k\sum_{j=1}^{n}\sum_{i=2}^{d}p(h_i=k,h_{i-1}=k'|\mathcal{D}_j;\theta_{\text{old}})}$
            3. $B_{m,k} = \frac{\sum_{j=1}^{n}\sum_{i=1}^{d}\mathbb{I}(v_i^{(j)}=m)p(h_i=k|\mathcal{D}_j;\theta_{\text{old}})}{\sum_{m}\sum_{j=1}^{n}\sum_{i=1}^{d}\mathbb{I}(v_i^{(j)}=m)p(h_i=k|\mathcal{D}_j;\theta_{\text{old}})}$
        
        Args: 
            data_loader:
                a list of lists whose elements are integers.
            hk_list:
                a list of np.arrays correspondding to the value of 
                $p(h_i | D_j; \theta_{old})$
            hkk_list:
                a list of np.arrays correspondding to the value of 
                $p(h_i, h_{i-1} | D_j; \theta_{old})$
            
        Returns:
            _initial_:
                1D array with shape (self.num_hiddens,) which specifies the 
                distribution over the initial hidden states
            _transition_:
                2D array with shape (self.num_hiddens, self.num_hiddens) where
                cell (j,k) specifies the following probability:
                    $p(h_{t-1}=j, h_{t}=k | D_j; \theta_{old})$
            _emission_:
                2D array with shape (self.num_observations, self.num_hiddens) 
                where cell (j,k) specifies the following probability:
                    $p(v_i=j | h_i=k, D_j; \theta_{old})$
        """
        
        _initial_ = np.zeros(self.num_hiddens)
        _transition_ = np.zeros([self.num_hiddens, self.num_hiddens]) 
        _emission_ = np.zeros([self.num_observations, self.num_hiddens])
        
        for j, obs in enumerate(data_loader):
            # Retrieve the distributions inferred in the E-step for the current observation obs
            hk = hk_list[j]
            hkk = hkk_list[j]
            # Handle obs of length 1 for which hkk is None
            hkk = hkk if hkk is not None else float('-inf')*np.ones([1, self.num_hiddens, self.num_hiddens])

            # TODO: your code here (keep the left hand side)
            _initial_ += np.exp(hk[0])
            _transition_ += np.exp(hkk).sum(axis=0)
            for m, ob in enumerate(obs):
                _emission_[ob] += np.exp(hk[m])
                #hint: not _emission_[obs] += np.exp(hk)

        # Normalise the distributions
        _initial_ /= len(data_loader)
        _transition_ /= _transition_.sum(axis=1, keepdims=True)
        _emission_ /= _emission_.sum(axis=0, keepdims=True)
        
        return _initial_, _transition_, _emission_
        
    @staticmethod
    def _stop_criterion(step:int=0, 
                        delta_param:float=1e-3,
                        delta_logpx:float=1e-1
                        ) -> bool:
        """ Criterion for stopping the iteration in Baum-Welch algorithm

        This method keeps tracking the number of steps and change of parameters/
        loglikelihood in order to check if the stop criterion has been 
        satisfied. If so, return True such that the Baum-Welch algorithm could
        stop. Otherwise, return False such that the Baum-Welch algorithm could
        keep going.
        
        Args:
            step:
                an integer indicating the index of the last iteration
            delta_param:
                a float indicating the change of parameters during the last
                iteration
            delta_logpx: 
                a float indicating the change of log-likelihood of the data
                during the last iteration
        
        Returns:
            bool:
                True for stopping the iteration, False for keeping it going.
        """
        max_steps = 100
        min_delta_param = 1e-16
        min_delta_logpx = 1e-8
        stop_condition = (step >= max_steps 
                        or delta_param < min_delta_param 
                        or delta_logpx < min_delta_logpx)
        return stop_condition
        
    def _initial_params(self) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Initialisation of the parameters
        
        This method would return the parameters for initialising a discrete 
        HMM model. 
        """
        initial = np.random.uniform(size=self.num_hiddens)
        initial /= initial.sum(axis=0)
        
        transition = np.random.uniform(size=[self.num_hiddens, 
                                             self.num_hiddens])
        transition /= transition.sum(axis=1, keepdims=True)
        
        emission = np.random.uniform(size=[self.num_observations, 
                                           self.num_hiddens])
        emission /= emission.sum(axis=0, keepdims=True)
        
        # TODO: initialise initial/transition to constants and randomly initialise emission, see what would happen
        
        return initial, transition, emission
    
    def baum_welch(self, data_loader:List[List[int]], verbose:bool=True):
        """Unsupervised learning method of discrete HMM model, i.e. Baum-Welch algorithm.
        
        This method would fit the parameters of an HMM model in an unsupervised 
        fashion. The overall procedure has been illustrated in the lecture and
        the HMM Learning notebook.
        
        In the following implementation, the overall framework has been 
        provided. The each step has also been commented below.
        
        Args:
            data_loader:
                a list whose elements are lists of integers.
            verbose:
                a boolean variable, print loglikelihood of each step if true.
        Returns:
            loglikelihood_list:
                a list of float, stores all the loglikelihood during the fitting
                procedure.
        """
        
        # Step 1: initialise the parameters for HMM model
        initial, transition, emission = self._initial_params()
        self.model = HMM(np.log(initial), 
                         np.log(transition), 
                         np.log(emission)
                         )
        
        # Step 2: set up the following variables for repeating the e/m-steps.
        stop = False                    # flag for stopping the loop
        step = 0                        # track the number of steps
        delta_param = math.inf          # track the change of parameters
        delta_loglikelihood = math.inf  # track the change of log-likelihood
        last_loglikelihood = 0.
        loglikelihood_list = []
        
        # Step 3: repeat the e/m-steps
        while not stop:
            # step 3.1: e-step
            loglikelihood, hk_list, hkk_list = self._e_step(data_loader)
            # step 3.2: m-step
            _initial_, _transition_, _emission_ = \
                self._m_step(data_loader, hk_list, hkk_list)
            
            # step 3.3: track step and change of parameters/log-likelihoods
            step += 1
            delta_param = self.model.get_delta_param(
                                        np.log(_initial_),
                                        np.log(_transition_),
                                        np.log(_emission_)
                                    )
            delta_loglikelihood = abs(loglikelihood - last_loglikelihood)
            last_loglikelihood = loglikelihood
            
            # step 3.4: update the parameters of HMM model
            self.model.initial = np.log(_initial_)
            self.model.transition = np.log(_transition_)
            self.model.emission = np.log(_emission_)
            
            #step 3.5: check if we're going to end the loop now
            stop = self._stop_criterion(step, delta_param, delta_loglikelihood)
            
            # monitor the learning procedure
            loglikelihood_list.append(loglikelihood)
            if verbose:
                print('step:', step, '\tloglikelihood:', loglikelihood)

        self._trained_ = True
        return loglikelihood_list
    
    def counts(self, data_loader:List) -> None:
        """Supervised learning method of HMM model, i.e. counting.
        
        This method would fit the parameters of an HMM model in a supervised 
        fashion where the optimisation problem is reduced to counting. 
        
        Args:
            data_loader:
                a list whose element should be a tuple containing two lists of integers.

        Returns:
            None
        """
        _initial_ = np.zeros(self.num_hiddens)
        _transition_ = np.zeros([self.num_hiddens, self.num_hiddens]) 
        _emission_ = np.zeros([self.num_observations, self.num_hiddens])
        
        for pair in data_loader:
            observations = pair[0]
            hiddens = pair[1]
            
            _initial_[hiddens[0]] += 1
            _emission_[observations[0]][hiddens[0]] += 1
            
            for i in range(1, len(hiddens)):
                _transition_[hiddens[i]][hiddens[i-1]] += 1
                _emission_[observations[i]][hiddens[i]] += 1
                
        _initial_ / len(data_loader)
        _transition_ = _transition_ / _transition_.sum(axis=0, keepdims=True)
        _emission_ = _emission_ / _emission_.sum(axis=0, keepdims=True)
        
        self.model = HMM(np.log(_initial_), 
                         np.log(_transition_), 
                         np.log(_emission_)
                         )
    
    def get_trained_model(self) -> HMM:
        """Return the HMM model with fitted parameters
        
        This method will first check if the model has been trained. If so, it 
        will return the model as an object of class HMM from hmm.py.
        
        Args:
            None
            
        Returns:
            self.model:
                an instance of HMM class from hmm.py
        """
        
        assert self.model is not None and self._trained_, \
            "The model has not been trained yet!"
        return self.model
