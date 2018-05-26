# coding: utf-8
from bs4 import BeautifulSoup
import os
import csv
import regex as re
from nltk import word_tokenize
from pymystem3 import Mystem
import pickle

MULTINUCLEAR_RELATIONS = ['comparison', 'contrast', 'joint', 'restatement', 'same-unit', 'sequence']
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


class EDU:
    def __init__(self, edu_text, position):
        self.text = edu_text
        self.position = int(position)
        self.tokens = word_tokenize(self.text.strip('#'))
        self.sentence_start = self.starts_sentence()
        self.sentence_end = self.ends_sentence()
        self.lemmatized_tokens = []
        self.pos = []
        self.gram = []
        self.tokenize_analyze()

    def tokenize_analyze(self):
        for token in self.tokens:
            lemma, pos, gr = self.lemmatize_pos_tag(token)
            self.lemmatized_tokens.append(lemma)
            self.pos.append(pos)
            self.gram.append(gr)

    def lemmatize_pos_tag(self, token):
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

    def ends_sentence(self):
        sentence_splitters = '.?!…'
        if self.text[-1] in sentence_splitters:
            return True
        return False

    def starts_sentence(self):
        if self.text[0].upper() == self.text[0]:
            return True
        return False

    def __repr__(self):
        return self.text, self.position


class EDUPair:
    def __init__(self, edu1, edu2, relation, text_id):
        self.edu1 = edu1
        self.edu2 = edu2
        self.relation = relation
        self.text_id = text_id


def ends_sentence(edu):
    sentence_splitters = '.?!…'
    if edu.text[-1] in sentence_splitters:
        return True
    return False


def multinuclear(group):
    # if group.attrs['type'] == 'multinuc':
    #     return True
    if group.attrs['relname'] in MULTINUCLEAR_RELATIONS:
        return True
    return False


def detect_parent(edu):
    if 'parent' in edu.attrs:
        return edu.attrs['parent']
    return None


def recursive_parent(root_parent_id, needed_edu_id, depth, groups):
    if depth == 0:
        return None
    for group in groups:
        if group.attrs['id'] == root_parent_id:
            current_group_parent_id = detect_parent(group)
            if current_group_parent_id == needed_edu_id:
                res_group = group
                return res_group
            elif current_group_parent_id is None:
                res_group = None
                return res_group
            else:
                return recursive_parent(current_group_parent_id, needed_edu_id, depth - 1, groups)


def detect_relation(edu1, edu2, groups):
    edu1_parent = detect_parent(edu1)
    edu2_parent = detect_parent(edu2)
    edu1_id = edu1.attrs['id']
    edu2_id = edu2.attrs['id']
    nuclearity = None
    found = True
    if edu1_parent is None and edu2_parent is None:
        relation = 'no_relation'
    else:
        if edu1_parent == edu2_id:
            relation = edu1.attrs['relname']
            nuclearity = 'SN'
        elif edu2_parent == edu1_id:
            relation = edu2.attrs['relname']
            nuclearity = 'NS'
        elif edu1_parent == edu2_parent:
            relation1 = edu1.attrs['relname']
            relation2 = edu2.attrs['relname']
            if relation1 != 'span' and relation2 == 'span':
                relation = relation1
                nuclearity = 'SN'
            elif relation2 != 'span' and relation1 == 'span':
                relation = relation2
                nuclearity = 'NS'
            elif relation1 == relation2:
                relation = relation1
                if relation in MULTINUCLEAR_RELATIONS:
                    nuclearity = 'M'
                else:
                    relation = 'no_relation'
                    nuclearity = None
            else:
                relation = 'no_relation'
                nuclearity = None

        else:
            nuclearity = None
            found = False
            for group in groups:
                if group.attrs['id'] == edu1_parent and detect_parent(group) == edu2_id:
                    relation = group.attrs['relname']
                    if multinuclear(group):
                        nuclearity = 'M'
                    else:
                        nuclearity = 'SN'
                    found = True
                    break
                elif group.attrs['id'] == edu2_parent and detect_parent(group) == edu1_id:
                    relation = group.attrs['relname']
                    if multinuclear(group):
                        nuclearity = 'M'
                    else:
                        nuclearity = 'NS'
                    found = True
                    break
                elif group.attrs['id'] == edu1_parent and group.attrs['id'] == edu2_parent:
                    relation = group.attrs['relname']
                    if relation in MULTINUCLEAR_RELATIONS:
                        nuclearity = 'M'
                    else:
                        print('2', edu1.text, edu2.text, relation)
                    if relation != 'span' and relation != 'antithesis':
                        found = True
                        break
    if not found:
        edu1_rec_parent = recursive_parent(edu1_parent, edu2_id, 5, groups)
        edu2_rec_parent = recursive_parent(edu2_parent, edu1_id, 5, groups)
        edu1_rec_same_parent = recursive_parent(edu1_parent, edu2_parent, 4, groups)
        edu2_rec_same_parent = recursive_parent(edu2_parent, edu1_parent, 4, groups)
        if edu1_rec_parent:
            relation = edu1_rec_parent.attrs['relname']
            nuclearity = 'SN'
            found = True
        elif edu2_rec_parent:
            relation = edu2_rec_parent.attrs['relname']
            nuclearity = 'NS'
            found = True
        elif edu1_rec_same_parent:
            relation = edu1_rec_same_parent['relname']
            if multinuclear(edu1_rec_same_parent):
                nuclearity = 'M'
            else:
                nuclearity = 'SN'
            if relation != 'span' and relation != 'antithesis':
                found = True
        elif edu2_rec_same_parent:
            relation = edu2_rec_same_parent['relname']
            if multinuclear(edu2_rec_same_parent):
                nuclearity = 'M'
            else:
                nuclearity = 'NS'
            if relation != 'span' and relation != 'antithesis':
                found = True
        else:
            relation = 'no_relation'
            nuclearity = None
    if not found:
        relation = 'no_relation'
        nuclearity = None

    detected_relation = '_'.join([i for i in [relation, nuclearity] if i is not None])
    return EDU(edu1.text, edu1.attrs['id']), EDU(edu2.text, edu2.attrs['id']), detected_relation


def generate_pairs_rs3(text_soup, text_id, window):
    pairs = []
    edus = text_soup.find_all('segment')
    groups = text_soup.find_all('group')
    for i in range(len(edus) - window):
        if ends_sentence(edus[i]):
            continue
        else:
            for j in range(1, window + 1):
                pairs.append(EDUPair(*detect_relation(edus[i], edus[i + j], groups) + (text_id,)))
                if ends_sentence(edus[i + j]):
                    break
    return pairs


def read_corpus(dir_path):
    corpus_soup = []
    for filename in os.listdir(dir_path):
        # print(filename)
        with open(os.path.join(dir_path, filename), 'rb') as file:
            raw = file.read()
            soup = BeautifulSoup(raw, 'lxml')
            corpus_soup.append(soup)
    return corpus_soup


def pairs_to_pickle(edupairs, dirpath='../..'):
    with open(os.path.join(dirpath, "all_pairs.pkl"), 'wb') as outfile:
        pickle.dump(edupairs, file=outfile)
    target = []
    for pair in edupairs:
        target.append(pair.relation)
    with open(os.path.join(dirpath, "all_target.pkl"), 'wb') as outfile:
        pickle.dump(target, file=outfile)


def pairs_to_csv(edupairs, filepath='../all_data.csv'):
    with open(filepath, 'w') as outfile:
        writer = csv.writer(outfile, delimiter=';', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['EDU1_pos', 'EDU1_text', 'EDU2_pos', 'EDU2_text', 'Text_id', 'Relation'])
        for pair in edupairs:
            writer.writerow([pair.edu1.position, pair.edu1.text, pair.edu2.position,
                             pair.edu2.text, pair.text_id, pair.relation])


def generate_matrix_all(window=5, dir_path='../../corpus_rs3/corpus'):
    text_soups = read_corpus(dir_path)
    all_pairs = []
    for i, text_soup in enumerate(text_soups):
        print(str(i + 1) + '/' + str(len(text_soups)))
        pairs = generate_pairs_rs3(text_soup, i, window)
        all_pairs.extend(pairs)
    # pairs_to_csv(all_pairs)
    pairs_to_pickle(all_pairs)


if __name__ == '__main__':
    generate_matrix_all()
