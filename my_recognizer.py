import warnings
from asl_data import SinglesData

def calculate_score(model, X, length):
    """ Calculate the score using model for a sequence X

    """
    try:
        score = model.score(X, length)
    except:
        score = float('-inf')

    return score

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
    # TODO implement the recognizer

    for X, length in test_set.get_all_Xlengths().values():
        probs = {word: score for word, score in [(word, calculate_score(model, X, length)) for word, model in models.items()] if score}

        probabilities.append(probs)
        guesses.append(max(probs, key=probs.get))

    return probabilities, guesses
