#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle

from decode import decode
from align_ibm1 import *
from BLEU_score import BLEU_score
from lm_train import lm_train

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Overall, it seems that each corresponding pair of reference sentences are similar, especially semantically, though there are noticeable differences.
For example, the first pair of sentences match exactly, except the google sentence contains a comma. There are cases though
where the google translation is slightly off i.e. 'member of parliament' in the hansard sentence getting translated to 'deputy' in the google translation. 
There are also cases where the sentences have a pretty different form but have a similar meaning i.e. ['i cannot imagine anything so ridiculous .', 'it is the height of ridicule']
In general the pairs of sentences are related very closely semantically but there tend to be slight differences in a few words or 
the form of the sentences.

Using more than 2 reference sentences would probably increase our BLEU scores, because more sentences means more opportunities 
for unigrams, bigrams and trigrams in our candidate sentence to appear in one of the reference sentences. This would increase
the numerator of the BLEU score and therefore increase the overall BLEU score.

In general, our BLEU scores are not very high. The highest BLEU score we achieve is 0.75, which is for the 22nd sentence
decoded using the 10k, 15k, and 30k alignment models. However the rest of the BLEU scores are less than or equal to 0.55,
meaning our decoder is not doing a good job in translation.

Some trends we may observe: 
In general, BLEU score for n = 2 and n = 3 are 0. There is only one case for n = 3 where 
the BLEU score in non-zero, and it was only 0.288. n = 2 has more cases with non-zero scores, and n = 2 scores are 
lower than the corresponding n = 1 scores. 

BLEU scores are often the same across all the alignment models for the same sentence, but there are a considerable number
of cases where the BLEU score for the 10k, 15k and 30k alignment models have the same BLEU scores, which are greater
than in the 1k case for the same sentence. So sometimes training on more than 1k sentences improves the quality of translation



Sentence i: [AM1k:[n=1, n=2, n=3], ... AM30k:[n=1, n=2, n=3]]
"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model
    """
    return lm_train(data_dir, language, fn_LM)

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model
    """
    return align_ibm1(data_dir, num_sent, max_iter, fn_AM)

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """





def main():
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """


    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    '''
    f = open("Task5.txt", 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    for i, AM in enumerate(AMs):

        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(...)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()
    '''
    scores = []
    LM = _getLM('/u/cs401/A2 SMT/data/Hansard/Training/', 'e', 'lme')
    AM1k, AM10k, AM15k, AM30k  = _getAM('/u/cs401/A2 SMT/data/Hansard/Training/', 1000, 5, 'am1k'), _getAM('/u/cs401/A2 SMT/data/Hansard/Training/', 10000, 5, 'am10k'), \
                                 _getAM('/u/cs401/A2 SMT/data/Hansard/Training/', 15000, 5, 'am15k'), _getAM('/u/cs401/A2 SMT/data/Hansard/Training/', 30000, 5, 'am30k')

    ams =[AM1k, AM10k, AM15k, AM30k]
    with open('/u/cs401/A2 SMT/data/Hansard/Testing/Task5.f') as fre, open('/u/cs401/A2 SMT/data/Hansard/Testing/Task5.e')  as enh, open('/u/cs401/A2 SMT/data/Hansard/Testing/Task5.google.e')  as eng:
        fsents, esents, esentsg = fre.readlines(), enh.readlines(), eng.readlines()

    for i in range(len(fsents)):
        scores_inner = []
        fsent, refs = " ".join(preprocess(fsents[i], 'f').split()[1:-1]), [esents[i].strip(), esentsg[i].strip()]
        for am in ams:
            decoded = decode(fsent, LM, am)
            print(decoded)
            print(refs)
            b1 = BLEU_score(decoded, refs, 1, True)
            b1_ = BLEU_score(decoded, refs, 1, False)
            b2 = BLEU_score(decoded, refs, 2, True)
            b2_ = BLEU_score(decoded, refs, 2, False)
            b3 = BLEU_score(decoded, refs, 3, True)


            b2 = b2 * (b1_ ** 0.5)               #get true bleue score

            b3 = b3 * (b1_ ** (1/3)) * ((b2_ ** 2) ** (1/3))

            scores_inner.append([b1, b2, b3])
        scores.append(scores_inner)

    with open('Task5.txt', 'w') as f:
        f.write(discussion)
        for item in scores:
            f.write("%s\n" % item)









main()



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
#     args = parser.parse_args()
#
#     main(args)
