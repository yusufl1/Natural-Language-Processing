import os, fnmatch
import numpy as np
import re
import string


dataDir = '/u/cs401/A3/data/'
UP, LEFT, UP_LEFT = 1, 2, 3

def Levenshtein(r, h):
    """                                                                         
    Calculation of WER with Levenshtein distance.                               
                                                                                
    Works only for iterables up to 254 elements (uint8).                        
    O(nm) time ans space complexity.                                            
                                                                                
    Parameters                                                                  
    ----------                                                                  
    r : list of strings                                                                    
    h : list of strings                                                                   
                                                                                
    Returns                                                                     
    -------                                                                     
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively
                                                                                

    """
    n, m = len(r), len(h)
    R, B = np.zeros((n + 1, m + 1)), np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 or j == 0:
                R[i, j] = max(i, j)

    for i in range(n+1):
        for j in range(m+1):
            if i == 0:
                B[i, j] = LEFT
            elif j == 0:
                B[i, j] = UP


    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dels = R[i - 1, j] + 1
            if r[i - 1] == h[j - 1]:
                plus = 0
            else:
                plus = 1
            subs = R[i - 1, j - 1] + plus
            ins = R[i, j - 1] + 1
            R[i, j] = min(dels, subs, ins)
            if R[i, j] == dels:
                B[i, j] = UP
            elif R[i, j] == ins:
                B[i, j] = LEFT
            else:
                B[i, j] = UP_LEFT
    bi, bj = n, m
    delz, subz, inz = 0, 0, 0
    while bi != 0 or bj != 0:
        if B[bi, bj] == UP:
            delz += 1
            bi -= 1
        elif B[bi, bj] == LEFT:
            inz += 1
            bj -= 1
        else:
            if r[bi - 1] != h[bj - 1]:
                subz += 1
            bi -= 1
            bj -= 1

    WER = R[i, j] / n
    return WER, subz, inz, delz

def wer(r, h):
    return Levenshtein(r, h)



if __name__ == "__main__":
    MFCCs = []
    punc = string.punctuation
    punc = re.sub('\[|\]', '', punc)
    p1 = "\!|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\=|\?|\@|\\|\^|\_|\`|\{|\||\}|\~|\<|\>"
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*.txt')
            MFCC = [speaker]
            for file in files:
                MFCC.extend([os.path.join(dataDir, speaker, file)])
            MFCCs.append(MFCC)

    gwers = []
    kwers = []
    gs, ks = [], []
    lnum = 0
    for MFCC in MFCCs:
        kaldi, ref, google = MFCC[1], MFCC[2], MFCC[3]
        with open(kaldi) as kaldi, open(ref) as ref, open(google) as google:
            ref = ref.readlines()
            kaldi = kaldi.readlines()
            google = google.readlines()
            for i in range(len(ref)):
                rline = ref[i].strip().lower()
                rline = re.sub(p1, '', rline)
                rline = re.sub('"', '', rline)

                kline = kaldi[i].strip().lower()
                kline = re.sub(p1, '', kline)
                kline = re.sub('"', '', kline)

                gline = google[i].strip().lower()
                gline = re.sub(p1, '', gline)
                gline = re.sub('"', '', gline)
                #


                WER, subz, inz, delz = Levenshtein(rline.split(), kline.split())
                k = "[{}] [Kaldi] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(MFCC[0], i, WER, subz, inz, delz)
                ks.append(k)
                kwers.append(WER)

                WER, subz, inz, delz = Levenshtein(rline.split(), gline.split())
                g = "[{}] [Google] [{}] [{}] S:[{}], I:[{}], D:[{}]".format(MFCC[0], i, WER, subz, inz, delz)
                gs.append(g)
                gwers.append(WER)
    with open('asrDiscussion.txt', 'w') as f:
        for i in range(len(gs)):
            f.write(gs[i] + '\n')
            f.write(ks[i] + '\n')
        f.write("Google mean: " + str(np.mean(gwers)))
        f.write("Google standard deviation: " + str(np.std(gwers)))
        f.write("Kaldi mean: " + str(np.mean(kwers)))
        f.write("Kaldi standard deviation: " + str(np.std(kwers)))
        f.write("One mistake the google system makes is that it often does not transcribe 'mm' or 'mhm' sounds. The google system\
        also sometimes transcribes 'i' as 'uh', 'uh' as 'the' and 'yeah' as 'why'.\
The Kaldi system sometimes transcribes 'on' as 'and', 'well' as 'why'. The Kaldi system also sometimes simply\
misses a word present in the reference transcription. The system also incorrectly transcribes bigrams sometimes, like\
'i know' as 'all my'")


