import numpy as np
import sys
import argparse
import os
import json
import string
import csv


def load_vars():
    first_p_prn = ['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    second_p_prn = ['you', 'your', 'yours', 'u', 'ur', 'urs']
    third_p_prn = ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
    future_vb = ["'ll", "will", "gonna"]
    slang_ac = ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao',
                'sml', 'btw',
                'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn',
                'bbs', 'cya', 'ez', 'f2f',
                'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol',
                'fml']

    with open('/u/cs401/Wordlists/Conjunct') as conjunct:
        ccs = conjunct.read()
    with open('/u/cs401/Wordlists/femaleFirstNames.txt') as f_first_names:
        f_f_names = f_first_names.read()
    with open('/u/cs401/Wordlists/lastNames.txt') as last_names:
        l_names = last_names.read()
    with open('/u/cs401/Wordlists/maleFirstNames.txt') as m_first_names:
        m_f_names = m_first_names.read()

    bg_norms = {}
    fst_line = True
    with open('/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv') as bg_norms_file:
        bg_reader = csv.reader(bg_norms_file, delimiter=',')
        for row in bg_reader:
            if fst_line:
                fst_line = False
                continue
            bg_norms[row[1]] = row[3:6]

    w_norms = {}
    first_line = True
    with open('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv') as w_norms_file:
        w_reader = csv.reader(w_norms_file, delimiter=',')
        for row in w_reader:
            if first_line:
                first_line = False
                continue
            w_norms[row[1]] = [row[2]] + [row[5]] + [row[8]]

    return first_p_prn, second_p_prn, third_p_prn, future_vb, slang_ac, ccs, f_f_names, l_names, m_f_names, bg_norms, w_norms


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.zeros(29)

    # load lists and dicts used to check features
    first_p_prn, second_p_prn, third_p_prn, future_vb, slang_ac, ccs, f_f_names, l_names, m_f_names, bg_norms, w_norms = load_vars()

    first_pp = 0
    second_pp = 0
    third_pp = 0
    cc = 0
    past_vb = 0                    #initialize features
    fut_vb = 0
    commas = 0
    c_nouns = 0
    p_nouns = 0
    adverbs = 0
    wh = 0
    slang = 0
    upper = 0
    total_token_length = 0
    punc = 0
    multipunc = 0
    eos = comment.count("\n")       # count sentences

    t_AoA = 0
    t_IMG = 0
    t_FAM = 0
    numb_bg = 0
                                    # initialize variables for norms
    AoA_list = []
    IMG_list = []
    FAM_list = []

    t_vms = 0
    t_ams = 0
    t_dms = 0
    num_w = 0

    vms_list = []
    ams_list = []
    dms_list = []




    for token in comment.split():
        if "/" in token and len(token.split("/")) == 2: #split tokens into word and tag
            word, tag = token.split("/")
        else:
            splitted = token.split("/")    #deal with words containing / like 9/11
            word = '/'.join(splitted[:-1])
            tag = splitted[-1]

        if word in first_p_prn:
            first_pp += 1
        if word in second_p_prn:
            second_pp += 1
        if word in third_p_prn:
            third_pp += 1
        if tag == 'cc' or word in ccs:
            cc += 1                                         #check each word for presence of features and increment
        if tag == 'vbd' or tag == 'vbn':                       #feature variables
            past_vb += 1
        if word in future_vb:
            fut_vb += 1
        if word == ',':
            commas += 1
        if word[0] in string.punctuation and word[-1] in string.punctuation and len(word) > 1:
            multipunc += 1

        if tag == 'nn' or tag == 'nns':
            c_nouns += 1
        if tag == 'nnp' or tag == 'nnps' or word in f_f_names or word in l_names or word in m_f_names:
            p_nouns += 1
        if tag == 'rb' or tag == 'rbr' or tag == 'rbs':
            adverbs += 1
        if tag == 'wdt' or tag == 'wp' or tag == 'wp$' or tag == 'wrb':
            wh += 1
        if word in slang_ac:
            slang += 1
        if len(word) >= 3 and word.isupper():
            upper += 1
        # average length of sentences
        if word not in string.punctuation:
            total_token_length += len(word)
        if word in string.punctuation:
            punc += 1
        if word in bg_norms:
            t_AoA += int(bg_norms[word][0])        #accumulate total norm values to get avg and make list of values
            t_IMG += int(bg_norms[word][1])        # to get std dev
            t_FAM += int(bg_norms[word][2])
            numb_bg += 1

            AoA_list.append(int(bg_norms[word][0]))
            IMG_list.append(int(bg_norms[word][1]))
            FAM_list.append(int(bg_norms[word][2]))
        if word in w_norms:
            t_vms += float(w_norms[word][0])
            t_ams += float(w_norms[word][1])
            t_dms += float(w_norms[word][2])
            num_w += 1

            vms_list.append(float(w_norms[word][0]))
            ams_list.append(float(w_norms[word][1]))
            dms_list.append(float(w_norms[word][2]))



    if len(comment.split()) != 0 and len(comment.split()) != punc:
        avg_token_len = (total_token_length) / (len(comment.split()) - punc)
    else:
        avg_token_len = 1
    if eos != 0:
        avg_sentence_len = len(comment.split()) / eos       #make sure we dont get any divide by 0 errors
    else:
        eos = 1
        avg_sentence_len = len(comment.split())
    if numb_bg != 0:
        avg_AoA = t_AoA / numb_bg
        avg_IMG = t_IMG / numb_bg
        avg_FAM = t_FAM / numb_bg
        sd_AoA = np.std(AoA_list)
        sd_IMG = np.std(IMG_list)
        sd_FAM = np.std(FAM_list)
    else:
        avg_AoA = 0
        avg_IMG = 0
        avg_FAM = 0
        sd_AoA = 0
        sd_IMG = 0
        sd_FAM = 0
    if num_w != 0:
        avg_vms = t_vms / num_w
        avg_ams = t_ams / num_w
        avg_dms = t_dms / num_w
        sd_vms = np.std(vms_list)
        sd_ams = np.std(ams_list)
        sd_dms = np.std(dms_list)
    else:
        avg_vms = 0
        avg_ams = 0
        avg_dms = 0
        sd_vms = 0
        sd_ams = 0
        sd_dms = 0

    #create feature array
    feats[0], feats[1], feats[2], feats[3], feats[4], feats[5] = first_pp, second_pp, third_pp, cc, past_vb, fut_vb
    feats[6], feats[7], feats[8], feats[9], feats[10], feats[11] = commas, multipunc, c_nouns, p_nouns, adverbs, wh
    feats[12], feats[13], feats[14], feats[15], feats[16], feats[17] = slang, upper, avg_sentence_len, avg_token_len, eos, avg_AoA
    feats[18], feats[19], feats[20], feats[21], feats[22], feats[23] = avg_IMG, avg_FAM, sd_AoA, sd_IMG, sd_FAM, avg_vms
    feats[24], feats[25], feats[26], feats[27], feats[28] = avg_ams, avg_dms, sd_vms, sd_ams, sd_dms
    return feats




def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    alt_features = np.load("/u/cs401/A1/feats/Alt_feats.dat.npy")     #load features
    cen_features = np.load("/u/cs401/A1/feats/Center_feats.dat.npy")
    lef_features = np.load("/u/cs401/A1/feats/Left_feats.dat.npy")
    ri_features = np.load("/u/cs401/A1/feats/Right_feats.dat.npy")

    j = 0
    for field in data:
        feat = extract1(field["body"].lower())
        id = field['id']
        cat = field['cat']
        i = 0
        with open(cat + "_IDs.txt") as ids:   #find index of id
            line = ids.readlines()
            i = line.index(id + '\r\n')


        if cat == "Left":      #append relevant features and then category to feature array
            feat = np.append(feat, lef_features[i], axis=None)
            feat = np.append(feat, [0])
        elif cat == "Center":
            feat = np.append(feat, cen_features[i], axis=None)
            feat = np.append(feat, [1])
        elif cat == "Right":
            feat = np.append(feat, ri_features[i], axis=None)
            feat = np.append(feat, [2])
        elif cat == "Alt":
            feat = np.append(feat, alt_features[i], axis=None)
            feat = np.append(feat, [3])
        feats[j] = feat
        j += 1




    np.savez_compressed( args.output, feats)





if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()


    main(args)

