import sys
import argparse
import os
import json, pprint
import html
import re
import string
import spacy

indir1001626957 = 'data';

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''
    TAG_MAP = [
        ".",
        ",",
        "-LRB-",
        "-RRB-",
        "``",
        "\"\"",
        "''",
        ",",
        "$",
        "#",
        "AFX",
        "CC",
        "CD",
        "DT",
        "EX",
        "FW",
        "HYPH",
        "IN",
        "JJ",
        "JJR",
        "JJS",
        "LS",
        "MD",
        "NIL",
        "NN",
        "NNP",
        "NNPS",
        "NNS",
        "PDT",
        "POS",
        "PRP",
        "PRP$",
        "RB",
        "RBR",
        "RBS",
        "RP",
        "SP",
        "SYM",
        "TO",
        "UH",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "WDT",
        "WP",
        "WP$",
        "WRB",
        "ADD",
        "NFP",
        "GW",
        "XX",
        "BES",
        "HVS",
        "_SP",
    ]

    modComm = comment
    if 1 in steps:            #replace newline chars
        modComm = comment.replace("\\n", '')

    if 2 in steps:        #remove html chars
        modComm = html.unescape(modComm)

    if 3 in steps:      # remove urls
        modComm = re.sub(r'http\S+', '', modComm)
        modComm = re.sub(r'www\S+', '', modComm)

    if 4 in steps: #split punctuation

        p = '(\d.|\w.)(\?+|!+|\.+|\,|"+|>+|#+|\$+|\%+|\&+|\(+|\)+|\++|\*+|\<+|\=+|\;+|\[+|\]+|\@+|\:+|\^+|\_+|\{+|\}+|\~+|\`+)'
        modComm = re.sub(p, r'\1 \2', modComm)

        pp = '(\?+|!+|\.+|,|"+|>+|#+|\$+|\%+|\&+|\(+|\)+|\++|\*+|\<+|\=+|\;+|\[+|\]+|\@+|\:+|\^+|\_+|\{+|\}+|\~+|\`+)(\d.|\w.)'
        modComm = re.sub(pp, r'\1 \2', modComm)



    if 5 in steps:   #clitics
        m = modComm.split("'")
        modComm = ''
        modComm += str(m[0])
        for i in range(1, len(m)):
            modComm += " '" + m[i]

    if 6 in steps: #POS tagging
        m = modComm

        utt = nlp(modComm)
        modComm = ''
        for token in utt:
            modComm += token.text + "/" + token.tag_ + " "



    if 7 in steps: #remove stopwords
        mc = ''
        with open('StopWords') as s:
            swords = s.readlines()
        for word in modComm.split():
            if word.split('/')[0].lower() not in swords:
                mc += word + " "
        modComm = mc

    if 8 in steps:      #lemmatize and deal with tagged words

        n = ''
        utt = nlp(modComm)
        for token in utt:
            if '/' not in token.text and token.text not in TAG_MAP:
                if not token.lemma_[0] == '-':
                    n += token.lemma_ + "/" + token.tag_ + " "
                else:
                    n += token.text + "/" + token.tag_ + " "
            elif token.text == './.':
                n += './. '
            elif token.text == ',':
                n += ',/, '
            elif token.text[0].isnumeric():
                n += token.text + " "
        modComm = n

    if 9 in steps:  #add newline chars to end of sentence
        mc = modComm.split()
        k = ''

        for word in mc:
            if word == './.':
                k += './. \n '
            elif word == '!':
                k += '!/. \n '
            elif word == '?':
                k += '?/. \n '
            else:
                k += str(word) + " "
        modComm = k


    if 10 in steps: #make lowercase
        modComm = modComm.lower()



    return modComm

def main( args ):
    print(args)

    allOutput = []

    for subdir, dirs, files in os.walk(indir1001626957):
        print(subdir, dirs, files)
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            cat = file

            start_i = int(args.ID[0]) % len(data)

            for i in range(start_i, start_i + int(args.max)):
                j = json.loads(data[i])
                id = j['id']
                body = j['body']
                body = preproc1(body)
                result = {}
                result['id'] = id
                result['body'] = body
                result['cat'] = cat
                allOutput.append(result)




    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":





    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (int(args.max) > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
