import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
import asl_utils


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # The logL values from the model increase, however, the formula for the Bayesian Information Criteria
        # makes this negative, so overall we have a minimization problem. Therefore, we start with positive infinity.
        best_BIC_score = math.inf
        best_model = None
        
        # The number of datapoints is given by the length of the array self.lengths which is a list
        # of how long all the sequences representing the words are.  
        num_datapoints = len(self.lengths)
        logN = math.log(num_datapoints)
        if self.verbose: print("Number of datapoints: ", num_datapoints)
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(num_states)
                logL = current_model.score(self.X, self.lengths)
                # Information taken from the forum. It is not entirely clear why this is the case
                # TODO: Need to verify. 
                num_parameters = num_states**2 + 2 * num_states * num_datapoints - 1
                current_BIC = -2 * logL + num_parameters * logN
                if self.verbose: print("Scores: BIC: {} logL: {} logN: {} P: {} N_Features: {}".format(current_BIC, logL, logN, num_parameters, current_model.n_features))
                if current_BIC < best_BIC_score:
                    best_BIC_score = current_BIC
                    best_model = current_model  
            except:
                if self.verbose: print("Error")
                continue
        
        print(self.this_word, " BIC ", best_BIC_score)      
        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_DIC_score = -math.inf
        best_model = None
        
        # In the paper mentioned above an important part was the parameter alpha which was used to
        # weigh the influence of the competing words. For the specific example in the paper, a lower
        # value of alpha performed significanlty better than values close to 1.
        # However, this was not given in the above formula for DIC
        alpha = 1.0
         
        if self.verbose: print("========= Now training word: ", self.this_word)
        
        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                current_model = self.base_model(num_states)
                current_model_log_score = current_model.score(self.X, self.lengths)
                if self.verbose: print("Score current_model_log_score for this model is: ", current_model_log_score)
                sum_competing_word_log_scores = 0
                for word in self.words:
                    if word != self.this_word:
                        try:
                            competing_X, competing_length = self.hwords[word]
                            competing_word_log_score = current_model.score(competing_X, competing_length)
                        except:
                            competing_word_log_score = 0
                        if self.verbose: print("Competing word: {} with score: {}".format(word, competing_word_log_score))
                        sum_competing_word_log_scores += competing_word_log_score
                current_DIC_score = current_model_log_score - alpha * sum_competing_word_log_scores / (len(self.words)-1)
                if self.verbose: print("Current DIC score: ", current_DIC_score)
                
                if current_DIC_score > best_DIC_score:
                    best_DIC_score = current_DIC_score
                    best_model = current_model  
            except:
                if self.verbose: print("Error")
                continue
              
        print("Best DIC", best_DIC_score, " with alpha ", alpha)      
        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_CV_score = -math.inf
        best_model = None

        split_data = KFold()

        for num_states in range(self.min_n_components, self.max_n_components + 1):
            try:
                # Only use cross-validation if we have enough datapoints to actually do splitting.
                # In the standard case n_splits is set to 3. 
                if len(self.lengths) > split_data.n_splits:
                    current_CV_score = 0
                    for cv_train_idx, cv_test_idx in split_data.split(self.sequences):
                        cv_train_X, cv_train_lengths = asl_utils.combine_sequences(cv_train_idx, self.sequences)
                        cv_test_X, cv_test_lengths = asl_utils.combine_sequences(cv_test_idx, self.sequences)
                        current_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(cv_train_X, cv_train_lengths)
                        current_CV_score += current_model.score(cv_test_X, cv_test_lengths)
                    current_CV_score = current_CV_score / split_data.n_splits
                    if self.verbose: print("Score for tests: ", current_CV_score)
                else:
                    # If we don't have enough data, we just use the log score
                    current_model = self.base_model(num_states)
                    current_CV_score = current_model.score(self.X, self.lengths)
               
                if current_CV_score > best_CV_score:
                    best_CV_score = current_CV_score
                    best_model = current_model  
            except:
                if self.verbose: print("Error")
                continue
              
        print("Best CV:", best_CV_score)      
        return best_model
