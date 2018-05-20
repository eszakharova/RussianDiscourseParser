# coding: utf-8
from __future__ import unicode_literals
from __future__ import division
from pymystem3 import Mystem
import regex as re
from nltk import word_tokenize

analyzer = Mystem()
pos_map = {'A': 'ADJ',
           'ADV': 'ADV',
           'ADVPRO': 'PRON',
           'ANUM': 'ADJ',
           'APRO': 'PRON',
           'COM': 'NOUN',
           'CONJ': 'CCONJ',
           'INTJ': 'PART',
           'NUM': 'NUM',
           'PART': 'PART',
           'PR': 'ADP',
           'S': 'NOUN',
           'SPRO': 'PRON',
           'V': 'VERB'}


def tokenize(string, need_lemmatization=False, need_analysis=False):
    if need_lemmatization:
        tokens = [lemmatize_analyze(i.lower(), need_analysis=need_analysis)
                      for i in word_tokenize(string)]
    else:
        tokens = [i.lower() for i in word_tokenize(string)]

    return tokens


def lemmatize_analyze(token, need_analysis):
    if need_analysis:
        analysis = analyzer.analyze(token)
        try:
            lemma = analysis[0]['analysis'][0]['lex']
            info = re.split('[,=]+?', analysis[0]['analysis'][0]['gr'])
            pos = pos_map[info[0]]
            gr = info[1:]
        except (KeyError, IndexError) as e:
            lemma = token
            pos = 'unknown'
            gr = []
        return lemma, pos, gr
    else:
        try:
            lemma = analyzer.lemmatize(token)[0]
        except (KeyError, IndexError) as e:
            lemma = token
        return lemma

def pseudoclause_split(text):
    pseudoclauses  = []
    return pseudoclauses
