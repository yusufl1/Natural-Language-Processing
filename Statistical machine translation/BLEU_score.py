import math
from preprocess import preprocess


def BLEU_score(candidate, references, n, brevity=False):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on

    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.

    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.


	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""

    for i in range(len(references)):
        references[i] = " ".join(preprocess(references[i], 'e').split()[1:-1])

    p1 = 0

    can = candidate.split()
    flatref, flatsent = [], ''

    for sent in references:
        flatref.extend(sent.split())
        flatsent += sent

    for word in can:
        if word in flatref:
            p1 += 1     #unigram precision

    p1 = p1 / len(can)

    if n >= 2:
        p2 = 0                              #bigram precision

        for i in range(len(can)):
            if i + 1 < len(can):

                if " ".join([can[i], can[i+1]]) in flatsent:
                    p2 += 1


        p2 /= (len(can) - 1)

    if n >= 3:
        p3 = 0                              #trigram precision

        for i in range(len(can)):
            if i + 2 < len(can):

                if " ".join([can[i], can[i+1], can[i+2]]) in flatsent:
                    p3 += 1

        p3 /= (len(can) - 2)

    if brevity:
        refs = []
        for sent in references:
            refs.append(sent.split())

        closest = 100
        for i in range(len(refs)):
            if abs(len(can) - len(refs[i])) < closest:
                closest = abs(len(can) - len(refs[i]))

        brev = closest / len(can)

        if brev < 1:
            bp = 1
        else:
            bp = math.exp(1 - brev)



    if n == 1 and brevity:
        bleu_score = bp * p1
    elif n == 2 and brevity:
        bleu_score = bp * ( p2) ** (1/n)
    elif n == 3 and brevity:
        bleu_score = bp * (( p3) ** (1 / n))
    elif n == 1 and not brevity:
        bleu_score = p1
    elif n == 2 and not brevity:
        bleu_score = ((p2) ** (1/n))
    elif n == 3 and not brevity:
        bleu_score = ((p3) ** (1 / n))






    return bleu_score

can = 'It is to insure the troops forever hearing the activity guidebook that party direct'
refs = [' It is a guide to action that ensures that the military will forever heed Party commands', 'It is the guiding principle which guarantees the military forces always being under command of the Party', 'It is the practical guide for the army always to heed the directions of the party']

#print(BLEU_score(preprocess(can, 'e'), refs, 2))
