from typing import Tuple
import numpy as np
import numpy.typing as npt


class IdentityDict(dict):
    """This special dictionary can map a number to itself, and is used in the 
    mappings of hidden states in the simulated data.
    """
    def __missing__(self, key):
        return key

class DataLoader(object):
    """Data loader for HMM models to infer/learn.

    This class provides list of observed sequences w/o the corresponding hidden 
    states. All strings will be UTF-8 encoded. The data has two kinds of 
    sources: 1) simulation; or 2) real world task Part-of-Speech (POS) tagging
    whose further description can be found at TODO. Format of the two types of
    data are illustrated as follows.
    
    Case 1: simulated data
        * observation: a list of integers, e.g. [1, 2, 3, 4, 0]. 
        * hidden: a list of integers, e.g. [0, 1, 2, 3, 4]. 

    Case 2: TODO

    Attributes:
        method: 
            a string indicating the source of data, could be "simulate" or
            "real"
        include_hiddens: 
            a boolen variable indicating whether hidden states will be returned 
            along with the observed sequences. 
        data_path: 
            a string indicting the path to the real world data set.
        initial: 
            a numpy array to specify the distribution over initial states, whose
            shape should be (num_hiddens,)
        transition: 
            a numpy array to specify the transition probability between
            hidden states, whose shape should be (num_hiddens, num_hiddens)
        emission: 
            a numpy array to specify the emission probability from hidden
            states to observations, whose shape should be 
            (num_observations, num_hiddents)
        D:
            an integer to fix the length of simulated sequences.
        num_hiddens:
            an integer, the number of different hiddent states.
        num_observations:
            an integer, the number of possible observations.
    Methods:
        TODO
        TODO:
            return the marginal probs of the generated sequences
    """
    
    def __init__(self, method:str='simulate', 
                include_hiddens:bool=False, 
                data_path:str=None,
                initial:npt.ArrayLike=None,
                transition:npt.ArrayLike=None,
                emission:npt.ArrayLike=None,
                D:int=None
                ) -> None:
        """Initialise the necessary parameters of DataLoader.
        
        This method initialises all necessary attributes of the class in order 
        to simulate data or read POS tagging data set.
        
        Args:
            meanings of arguments are illustrated in the comments to class
            attributes.
            
            Note that there are two mutually exclusive scenarios:
             1. method is 'simulate':
                initial, transition, and emission all need to be specified in 
                order to simulate data
             2. method is 'pos':
                data_path needs to be specified in order to read data
            
        Returns:
            None
        """
        super().__init__()
        
        # check the value of method
        assert method in ['simulate', 'pos'], \
            "Unknown source of data: {0}; please input either \'simulate\' or \
                \'pos\'.".format(method)
        self.method = method
        self.include_hiddens = include_hiddens
        
        if method == 'simulate':
            assert (initial is not None) and (transition is not None) and \
                (emission is not None), \
                "Please check the specified \'initial\', \'transition\', and" \
                    + " \'emission\'."

            assert (initial.shape[0] == transition.shape[1]) and \
                (transition.shape[0] == transition.shape[1]) and \
                    (transition.shape[0] == emission.shape[1]), \
                "Please check the dimensions of  \'initial\', \'transition\',"\
                    + " and \'emission\'."
        
            self.initial = initial
            self.transition = transition
            self.emission = emission
            self.D = D
            
            self.num_hiddens = self.transition.shape[0]
            self.num_observations = self.emission.shape[0]
            
        else:
            assert data_path is not None, \
                "Please specify the path to the POS tagging dataset."
            
            self.data_path = data_path
            # TODO: complete the pos tagging data set loader.
            
        
        self._construct_idx2str()
    
    def _construct_idx2str(self) -> None:
        """Setup the mappings from indices to hiddens/observations.
        
        This method constructs the mappings from the integers to the capital
        alphabets that will appear in the observations.
        
        Args: 
            None
            
        Returns:
            None
        """
        
        if self.method == 'simulate':
            self.hidden_dict = IdentityDict()
            self.observation_dict = IdentityDict()
        else:
            # TODO
            # create the dictionaries used in real data set
            pass
        
    def _generate_data(self):
        """Generate a sequence of hidden states and the corresponding
        observations.
        
        This method will first sample a sequence of hiddent states following
        the Markov chain specified by self.initial and self.transmition. Then
        it will sample observations for each hidden state following 
        self.emission.
        
        Args:
            None
            
        Returns:
            A **list** consists of the following two elements:
            observations:
                a list of integers
            hiddens:
                a list of integers
        """
        
        hidden_states = []
        observations = []

        end = 0
        last_transition = self.initial
        step = 0
        
        while not end:
            # TODO: dig hole here
            # sample current hidden state
            cur_h = np.random.choice(self.num_hiddens, 1, p=last_transition)[0]
            # sample current observation
            cur_o = np.random.choice(self.num_observations, 1,
                                     p=self.emission[:, cur_h])[0]
            
            # add the sampled hidden state and observation to the lists
            hidden_states.append(cur_h)
            observations.append(cur_o)
            
            step += 1
            # sample if the next hiddent state is the end of sequence
            end = (cur_o == 0) if self.D is None else int(step >= self.D)
            last_transition = self.transition[cur_h, :]
            
        hidden_states = [self.hidden_dict[x] for x in hidden_states]
        observations = [self.observation_dict[x] for x in observations]
        
        return observations, hidden_states
    
    def _read_postag_data(self):
        pass
    
    def get_true_params(self) -> \
        Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """Return a tuple consists of the true parameters of data generation.
        
        This method returns a tuple consisits of the three parameters of the HMM
        that genreates the data list.
        
        Args:
            None
            
        Returns:
            self.initial: 
                an numpy.array which specifies the distribution over initial
                states.
            self.transition: 
                an numpy.array which specifies the transition probability between hidden states.
            self.emission: 
                an numpy.array which specifies the emission probability from hidden states to observations.
            self.end:
                an numpy.array which specifies the transition probability from 
                hidden states to end of sequence.
        """
        if self.method == 'pos':
            print("Oops, we don't know the true parameters of the real-world \
                dataset.")
            return None
        else:
            return self.initial, self.transition, self.emission, self.end
    
    def get_data_list(self, n_samples:int=None):
        """Return a list of sequences as sample for HMM's inference/learning.
        
        This method returns a list of (lists of integers) to be the inputs for 
        the inference/learning modules of HMM. There will be two kinds of lists
        according to different sources of data:
        1) 'simulate': 
            * observation: a list of integers, e.g. [0, 1, 2, 1, 3];
            * hidden: a list of integers, e.g. [0, 1, 2, 1, 3].
        2) 'pos': in this case, each string would be like TODO
        
        Args:
            n_samples:
                Optional, an integer to specify the number of samples in the 
                simulated data list. 
                self.method should be 'simulate' if this argument is given.
        
        Returns:
            data_list:
                a list consists of 
        """
        
        data_list = []
        
        if self.method == 'simulate':
            assert type(n_samples) == int, \
                "Please specify the number of sample you want to generate."
            for _ in range(n_samples):
                if self.include_hiddens:
                    data_list.append(self._generate_data())
                else:
                    data_list.append(self._generate_data()[0])
        else:
            pass # TODO
            
        return data_list
    
