import re


def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence

	OUTPUT:
	out_sentence: (string) the modified sentence
    """
    sent = in_sentence.lower()

    if language == 'f':
        out_sentence = re.findall(r"(?:l'|c'|t'|qu'|j'|lorsqu'|puisqu')|[\w]+|[.,?!;\"/><(){}\[\]\*\&\^\-\+]", sent)
    elif language == 'e':
        out_sentence = re.findall(r"[\w]+|[,.?!;\"/><()}{\[\]\*\&\^\+\-]", sent)

    out_sentence = ["SENTSTART"] + out_sentence + ["SENTEND"]
    out_sentence = " ".join(out_sentence)


    return out_sentence


