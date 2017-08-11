import warnings
import math
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    for index in range(0, test_set.num_items):
        test_X, test_length = test_set.get_item_Xlengths(index)
        best_guess = None
        best_score = -math.inf
        print("Now evaluating scores for word: ", test_set.wordlist[index])
        probability_dict = {}
        for current_word, model in models.items():
            try:
                current_score = model.score(test_X, test_length)
            except:
                current_score = -math.inf
            probability_dict[current_word] = current_score
            # print("Score for {} is {}".format(current_word, current_score))
            
            if current_score > best_score:
                best_score = current_score
                best_guess = current_word

        print("Best guess: ", best_guess, " with score: ", best_score)
        probabilities.append(probability_dict)
        guesses.append(best_guess)
    
    return (probabilities, guesses)
