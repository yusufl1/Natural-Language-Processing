from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    log_prob = 0

    sent = sentence.split()

    for i in range(len(sent)):
        if i + 1 < len(sent):
            if sent[i + 1] in LM['bi'][sent[i]] and sent[i] in LM["uni"]:
                if smoothing == True:
                    num = LM['bi'][sent[i]][sent[i + 1]] + delta
                    denom = LM['uni'][sent[i]] + (delta * vocabSize)

                else:
                    num = LM['bi'][sent[i]][sent[i + 1]]
                    denom = LM['uni'][sent[i]]
                if num == 0 or denom == 0:
                    log_prob += 0
                else:
                    log_prob += log(num/denom, 2)


    return log_prob
