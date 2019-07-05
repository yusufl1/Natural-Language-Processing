from lm_train import *
from log_prob import *
from preprocess import preprocess
from math import log
import os
import pickle

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
    Implements the training of IBM-1 word alignment algoirthm.
    We assume that we are implemented P(foreign|english)

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider
    max_iter : 		(int) the maximum number of iterations of the EM algorithm
    fn_AM : 		(string) the location to save the alignment model

    OUTPUT:
    AM :			(dictionary) alignment model structure

    The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
    is the computed expectation that the foreign_word is produced by english_word.

            LM['house']['maison'] = 0.5
    """
    AM = {}

    # Read training data
    e_sents, f_sents = read_hansard(train_dir, num_sentences)


    # Initialize AM uniformly
    AM, AMcounts = initialize(e_sents, f_sents)



    # Iterate between E and M steps

    for i in range(max_iter):
        AM = em_step(AM, e_sents, f_sents)

    AM['SENTSTART'], AM['SENTEND'] = {}, {}
    AM['SENTSTART']['SENTSTART'], AM['SENTEND']['SENTEND'] = 1, 1


    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM

# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
    Read up to num_sentences from train_dir.

    INPUTS:
    train_dir : 	(string) The top-level directory name containing data
                    e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
    num_sentences : (int) the maximum number of training sentences to consider


    Make sure to preprocess!
    Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

    Make sure to read the files in an aligned manner.
    """
    e_sents = []
    f_sents = []

    for file in os.listdir(train_dir):
        if len(e_sents) < num_sentences and file[-1] == 'e' and train_dir[-1] == '/':
            with open(train_dir + file) as e_f, open(train_dir + file[:-1] + 'f') as f_f:
                sents_e, sents_f = e_f.readlines(), f_f.readlines()

                for i in range(len(sents_e)):
                    if len(e_sents) < num_sentences:
                        e_prcd = preprocess(sents_e[i].strip(), 'e')
                        e_prcd = e_prcd.split()
                        e_sents.append(" ".join(e_prcd[1:-1]))

                        f_prcd = preprocess(sents_f[i].strip(), 'f')
                        f_prcd = f_prcd.split()
                        f_sents.append(" ".join(f_prcd[1:-1]))
                    else:
                        break
        elif len(e_sents) < num_sentences and file[-1] == 'e' and train_dir[-1] != '/':
            with open(train_dir + '/' + file) as e_f, open(train_dir + '/' + file[:-1] + 'f') as f_f:
                sents_e, sents_f = e_f.readlines(), f_f.readlines()

                for i in range(len(sents_e)):
                    if len(e_sents) < num_sentences:
                        e_prcd = preprocess(sents_e[i].strip(), 'e')
                        e_prcd = e_prcd.split()
                        e_sents.append(" ".join(e_prcd[1:-1]))

                        f_prcd = preprocess(sents_f[i].strip(), 'f')
                        f_prcd = f_prcd.split()
                        f_sents.append(" ".join(f_prcd[1:-1]))
                    else:
                        break
        else:
            break

    return e_sents, f_sents



def initialize(eng, fre):
    """
    Initialize alignment model uniformly.
    Only set non-zero probabilities where word pairs appear in corresponding sentences.
    """

    AM = {}

    for i in range(len(eng)):
        esent, fsent = eng[i].split(), fre[i].split()
        for eng_word in esent:
            if eng_word not in AM:
                AM[eng_word] = {}
            for f_word in fsent:
                if f_word not in AM[eng_word]:
                    AM[eng_word][f_word] = 1
                else:
                    AM[eng_word][f_word] += 1
    AMu = {}

    for word in AM:
        AMu[word] = {}
        for entry in AM[word]:
            AMu[word][entry] = 1 / len(AM[word])

    return AMu, AM



def em_step(t, eng, fre):
    """
    One step in the EM algorithm.
    Follows the pseudo-code given in the tutorial slides.
    """
    tcount = {}
    total = {}

    for i in range(len(eng)):
        esent, fsent = eng[i].split(), fre[i].split()
        for word in esent:
            total[word] = 0
            if word not in tcount:
                tcount[word] = {}
            for fword in fsent:
                tcount[word][fword] = 0

    for i in range(len(eng)):
        esentu, fsentu = set(eng[i].split()), set(fre[i].split())
        esent, fsent = eng[i].split(), fre[i].split()

        for fwu in fsentu:
            denom_c = 0
            for ewu in esentu:
                denom_c += t[ewu][fwu] * fsent.count(fwu)
            for ew in esentu:
                tcount[ew][fwu] += t[ew][fwu] * fsent.count(fwu) * esent.count(ew) / denom_c
                total[ew] += t[ew][fwu] * fsent.count(fwu) * esent.count(ew) / denom_c

    for e in total:
        for f in tcount[e]:
            t[e][f] = tcount[e][f] / total[e]

    return t




