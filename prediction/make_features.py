import pandas as pd
import numpy as np
import pickle
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import classification_report, confusion_matrix
import regex as re
from collections import Counter
from imblearn.over_sampling import SMOTE
from gensim.models import Doc2Vec, Word2Vec
from collections import defaultdict


model_d = Doc2Vec.load('vec/model_d.w2v')
model_w = Word2Vec.load('vec/model_w.w2v')

# def iterate_pos(pos_list):
#     for pos in pos_list:
#         yield pos

def iterate_pos(pos_list):
    return pos_list



def rel_type_only(y):
    return [re.sub('_(SN|NS|M)', '', elem) for elem in y]


def nucl_only(y):
    return [elem.split('_')[1] if elem != 'no_relation' else elem for elem in y]


def presence_only(y):
    return ['relation' if elem != 'no_relation' else elem for elem in y]


def all_classification_reports(y_true, y_pred):
    print('All together:')
    print(classification_report(y_true, y_pred))
    print('Relation type:')
    print(classification_report(rel_type_only(y_true), rel_type_only(y_pred)))
    print('Nuclearity:')
    print(classification_report(nucl_only(y_true), nucl_only(y_pred)))
    print('Presence of relation:')
    print(classification_report(presence_only(y_true), presence_only(y_pred)))
    print('Confusion matrix:')
    print(confusion_matrix(y_true, y_pred))


def load_pairs_target_remove_minor_classes():
    with open('../all_pairs.pkl', 'rb') as file:
        pairs = pickle.load(file)
    with open('../all_target.pkl', 'rb') as file:
        target = pickle.load(file)
    ok_pairs = []
    ok_target = []
    value_count_dict = Counter(target)
    for i in range(len(target)):
        if value_count_dict[target[i]] > 2:
            ok_pairs.append(pairs[i])
            ok_target.append(target[i])
    return ok_pairs, ok_target


def load_pairs_target():
    with open('../all_pairs.pkl', 'rb') as file:
        pairs = pickle.load(file)
    with open('../all_target.pkl', 'rb') as file:
        target = pickle.load(file)
    ok_pairs = []
    ok_target = []
    value_count_dict = Counter(target)
    for i in range(len(target)):
        if value_count_dict[target[i]] > 1:
            ok_pairs.append(pairs[i])
            ok_target.append(target[i])
    return ok_pairs, ok_target


def has_markers(lemmas):
    result_dict = defaultdict(int)
    markers_dict = {'attribution': ['надеяться', 'опасаться', 'отметить', 'отмечать', 'сообщаться', 'утверждать', 'заявить', 'заявлять', 'передать', 'передавать', 'подчеркнуть', 'подчеркивать',
    'написать', 'рассказать', 'рассказывать', 'сообщить', 'сообщать', 'сказать', 'сообщать',
    'сообщаться', 'сообщить', 'писать', 'написать', 'объявить', 'объявлять'],
    'cause-effect': ['поскольку', 'причина', 'результат', 'вследствие', 'из-за', 'потому'],
    'concession': ['несмотря'],
    'condition': ['пока', 'если'],
    'contrast': ['вместо'],
    'elaboration': ['который'],
    'joint': ['параллельно'],
    'purpose' :['чтобы', 'для']}
    for rel in markers_dict:
        for marker in markers_dict[rel]:
            if marker in lemmas:
                result_dict[rel] = 1
                break
    return result_dict


    return
def generate_feature_matrix(pairs):
    c_vect = CountVectorizer(min_df=5, ngram_range=(1,3), tokenizer=word_tokenize)
    pos_vect = CountVectorizer(tokenizer=iterate_pos, ngram_range=(1,3), lowercase=False)
    DataDict = {'edu1_position': [],
                'edu2_position': [],
                'edu1_endsent': [],
                'edu1_startsent': [],
                'edu2_endsent': [],
                'edu2_startsent': [],
                'edu1_len': [],
                'edu2_len': [],
                'same_tokens': [],
                'attribution1': [],
                'cause-effect1': [],
                'concession1': [],
                'condition1': [],
                'contrast1': [],
                'elaboration1': [],
                'joint1': [],
                'purpose1' :[],
                'attribution2': [],
                'cause-effect2': [],
                'concession2': [],
                'condition2': [],
                'contrast2': [],
                'elaboration2': [],
                'joint2': [],
                'purpose2' :[]}
    for pair in pairs:
        markers_dict1 = has_markers(pair.edu1.lemmatized_tokens)
        markers_dict2 = has_markers(pair.edu2.lemmatized_tokens)
        DataDict['edu1_position'].append(int(pair.edu1.position))
        DataDict['edu2_position'].append(int(pair.edu2.position))
        DataDict['edu1_endsent'].append(int(pair.edu1.sentence_end))
        DataDict['edu2_endsent'].append(int(pair.edu2.sentence_end))
        DataDict['edu1_startsent'].append(int(pair.edu1.sentence_start))
        DataDict['edu2_startsent'].append(int(pair.edu2.sentence_start))
        DataDict['edu1_len'].append(len(pair.edu1.tokens))
        DataDict['edu2_len'].append(len(pair.edu2.tokens))
        # количество совпадающих токенов (леммы)
        DataDict['same_tokens'].append(len(set(pair.edu1.lemmatized_tokens).intersection(
            pair.edu2.lemmatized_tokens)))
        for rel_name in ['attribution','cause-effect','concession','condition','contrast','elaboration','joint','purpose']:
            DataDict[rel_name+'1'].append(markers_dict1[rel_name])
            DataDict[rel_name+'2'].append(markers_dict2[rel_name])


    X = pd.DataFrame(DataDict)

    # векторайзер по словам
    all_texts = [pair.edu1.text for pair in pairs] + [pair.edu2.text for pair in pairs]
    c_vect.fit(all_texts)
    edus1_vect = c_vect.transform([pair.edu1.text for pair in pairs])
    edus2_vect = c_vect.transform([pair.edu2.text for pair in pairs])

    # векторайзер по тегам частей речи
    all_pos = [pair.edu1.pos for pair in pairs] + [pair.edu2.pos for pair in pairs]
    pos_vect.fit(all_pos)
    pos1_vect = pos_vect.transform([pair.edu1.pos for pair in pairs])
    pos2_vect = pos_vect.transform([pair.edu2.pos for pair in pairs])

    # Doc2Vec - вектор ЭДЕ
    d2v1 = csr_matrix(np.array([model_d.infer_vector(pair.edu1.tokens) for pair in pairs]))
    d2v2 = csr_matrix(np.array([model_d.infer_vector(pair.edu1.tokens) for pair in pairs]))

    # Word2Vec - векторы первого и последнего токена в ЭДЕ
    w2v1_first = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu1.tokens[0]) for pair in pairs]))
    w2v2_first = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu2.tokens[0]) for pair in pairs]))
    w2v1_last = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu1.tokens[-1]) for pair in pairs]))
    w2v2_last = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu2.tokens[-1]) for pair in pairs]))

    X_sparse = csr_matrix(np.array(X))
    X_concat = hstack((X_sparse, edus1_vect, edus2_vect, pos1_vect, pos2_vect, d2v1, d2v2,
                       w2v1_first, w2v2_first, w2v1_last, w2v2_last))
    print(X_concat.shape)
    return X_concat

def generate_feature_matrix_(pairs):
    c_vect = CountVectorizer(min_df=5, tokenizer=word_tokenize)
    pos_vect = CountVectorizer(tokenizer=iterate_pos, lowercase=False)
    DataDict = {'edu1_position': [],
                'edu2_position': [],
                'edu1_endsent': [],
                'edu1_startsent': [],
                'edu2_endsent': [],
                'edu2_startsent': [],
                'edu1_len': [],
                'edu2_len': [],
                'same_tokens': []}
    for pair in pairs:
        DataDict['edu1_position'].append(int(pair.edu1.position))
        DataDict['edu2_position'].append(int(pair.edu2.position))
        DataDict['edu1_endsent'].append(int(pair.edu1.sentence_end))
        DataDict['edu2_endsent'].append(int(pair.edu2.sentence_end))
        DataDict['edu1_startsent'].append(int(pair.edu1.sentence_start))
        DataDict['edu2_startsent'].append(int(pair.edu2.sentence_start))
        DataDict['edu1_len'].append(len(pair.edu1.tokens))
        DataDict['edu2_len'].append(len(pair.edu2.tokens))
        # количество совпадающих токенов (леммы)
        DataDict['same_tokens'].append(len(set(pair.edu1.lemmatized_tokens).intersection(
            pair.edu2.lemmatized_tokens)))

    X = pd.DataFrame(DataDict)

    # векторайзер по словам
    all_texts = [pair.edu1.text for pair in pairs] + [pair.edu2.text for pair in pairs]
    c_vect.fit(all_texts)
    edus1_vect = c_vect.transform([pair.edu1.text for pair in pairs])
    edus2_vect = c_vect.transform([pair.edu2.text for pair in pairs])

    # векторайзер по тегам частей речи
    all_pos = [pair.edu1.pos for pair in pairs] + [pair.edu2.pos for pair in pairs]
    pos_vect.fit(all_pos)
    pos1_vect = pos_vect.transform([pair.edu1.pos for pair in pairs])
    pos2_vect = pos_vect.transform([pair.edu2.pos for pair in pairs])

    X_sparse = csr_matrix(np.array(X))
    X_concat = hstack((X_sparse, edus1_vect, edus2_vect, pos1_vect, pos2_vect))
    print(X_concat.shape)
    return X_concat

def w2v_word_vector(model, word):
    try:
        res = model.wv[word]
    except KeyError:
        res = np.zeros(100)
    return res


def smote_oversampling(X, y):
    sm = SMOTE(random_state=669, k_neighbors=1)
    X_res, y_res = sm.fit_sample(X, y)
    print(X_res.shape)
    return X_res, y_res


def smote_oversampling5(X, y):
    sm = SMOTE(random_state=669, k_neighbors=5)
    X_res, y_res = sm.fit_sample(X, y)
    print(X_res.shape)
    return X_res, y_res
