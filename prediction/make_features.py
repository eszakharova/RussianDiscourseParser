import pandas as pd
import numpy as np
import pickle
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import classification_report, confusion_matrix
import regex as re
from collections import Counter
from imblearn.over_sampling import SMOTE
from gensim.models import Doc2Vec, Word2Vec
from collections import defaultdict
from nltk.parse.malt import MaltParser
import matplotlib.pyplot as plt


def iterate_pos(pos_list):
    return pos_list


def rel_type_only(y):
    return [re.sub('_(SN|NS|M)', '', elem) for elem in y]


def nucl_only(y):
    return [elem.split('_')[1] if elem != 'no_relation' else elem for elem in y]


def presence_only(y):
    return ['relation' if elem != 'no_relation' else elem for elem in y]

def plot_normalized_confusion_matrix_colors(y_true, y_pred, title='Confusion matrix'):
    conf_arr = confusion_matrix(y_true, y_pred)
    labels = ['attribution_NS', 'attribution_SN', 'background_NS', 'background_SN', 'cause-effect_NS', 'cause-effect_SN',
        'comparison_M', 'concession_NS', 'concession_SN', 'condition_NS', 'condition_SN', 'contrast_M', 'elaboration_NS',
        'elaboration_SN', 'evidence_NS', 'evidence_SN', 'interpretation-evaluation_NS', 'interpretation-evaluation_SN',
        'joint_M', 'no_relation', 'preparation_SN', 'purpose_NS', 'purpose_SN', 'restatement_M', 'same-unit_M',
        'sequence_M']
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure('confusion matrix',figsize=(12, 12))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap='binary',
                    interpolation='nearest')
    width, height = conf_arr.shape
    plt.colorbar(res)
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(width), labels[:width], rotation='vertical')
    plt.yticks(range(height), labels[:height])

def plot_confusion_matrix_numbers(y_true, y_pred, title='Confusion matrix'):
    conf_arr = confusion_matrix(y_true, y_pred)
    labels = ['attribution_NS', 'attribution_SN', 'background_NS', 'background_SN', 'cause-effect_NS', 'cause-effect_SN',
        'comparison_M', 'concession_NS', 'concession_SN', 'condition_NS', 'condition_SN', 'contrast_M', 'elaboration_NS',
        'elaboration_SN', 'evidence_NS', 'evidence_SN', 'interpretation-evaluation_NS', 'interpretation-evaluation_SN',
        'joint_M', 'no_relation', 'preparation_SN', 'purpose_NS', 'purpose_SN', 'restatement_M', 'same-unit_M',
        'sequence_M']
    fig = plt.figure('confusion matrix', figsize=(12, 12))
    plt.clf()
    ax = fig.add_subplot(111)
    width, height = conf_arr.shape
    offset = .5
    ax.set_xlim(-offset, width - offset)
    ax.set_ylim(-offset, height - offset)
    ax.hlines(y=np.arange(height+1)- offset, xmin=-offset, xmax=width-offset)
    ax.vlines(x=np.arange(width+1) - offset, ymin=-offset, ymax=height-offset)

    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_arr[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')
    plt.title(title, fontsize=14)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(range(width), labels[:width], rotation='vertical')
    plt.yticks(range(height), labels[:height])


def all_classification_reports(y_true, y_pred):
    print('All together:')
    print(classification_report(y_true, y_pred))
    print('Relation type:')
    print(classification_report(rel_type_only(y_true), rel_type_only(y_pred)))
    print('Nuclearity:')
    print(classification_report(nucl_only(y_true), nucl_only(y_pred)))
    print('Presence of relation:')
    print(classification_report(presence_only(y_true), presence_only(y_pred)))
    # print('Confusion matrix:')
    # print(confusion_matrix(y_true, y_pred))


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
    markers_dict = {'attribution': ['надеяться', 'опасаться', 'отметить', 'отмечать', 'сообщаться',
    'утверждать', 'заявить', 'заявлять', 'передать', 'передавать', 'подчеркнуть', 'подчеркивать',
    'написать', 'рассказать', 'рассказывать', 'сообщить', 'сообщать', 'сказать', 'сообщать',
    'сообщаться', 'сообщить', 'писать', 'написать', 'объявить', 'объявлять', 'добавить', 'данные',
    'подтверждать', 'подтвердить'],
    'cause-effect': ['поскольку', 'причина', 'результат', 'вследствие', 'из-за', 'следовательно'
    'приводить', 'привести'],
    'concession': ['несмотря', 'хотя', 'даже'],
    'condition': ['пока', 'если'],
    'contrast': ['вместо', 'но', 'однако', 'несмотря', 'иначе', 'напротив'],
    'elaboration': ['который', 'особенно', 'включить', 'включать', 'где', 'например'],
    'evidence': ['поэтому', 'ведь', 'подтверждать', 'подтвердить', 'оказаться', 'оказываться'],
    'interpretation-evaluation': ['безусловно', 'выбрать', 'выбирать'],
    'joint': ['параллельно', 'либо', 'также', 'и'],
    'purpose': ['чтобы', 'для', 'ради'],
    'sequence': ['позднее', 'позже', 'впоследствии']}
    for rel in markers_dict:
        for marker in markers_dict[rel]:
            if marker in lemmas:
                result_dict[rel] = 1
                break
    lem_str = ' '.join(lemmas)
    if 'потому что' in lem_str:
        result_dict['cause-effect'] = 1
    if ('потому' in lem_str and 'потому что' not in lem_str) or ('это показывать' in lem_str) or ('это показать' in lem_str) or ('это доказывать' in lem_str) or ('это доказать' in lem_str) or ('говорить о' in lem_str) or ('так как' in lem_str):
        result_dict['evidence'] = 1
    if 'при условие' in lem_str or 'в случай' in lem_str or 'в этот случай' in lem_str or 'в такой случай' in lem_str or 'лишь тогда' in lem_str:
        result_dict['condition'] = 1
    if 'иначе говорить' in lem_str or 'иной слова' in lem_str or 'то есть' in lem_str or 'то быть' in lem_str:
        result_dict['restatment'] = 1
    if 'в особенность' in lem_str or 'в частность' in lem_str or 'один из' in lem_str:
        result_dict['elaboration'] = 1
    if 'выдвинуть мнение' in lem_str or 'выдвигать мнение' in lem_str or 'по данные' in lem_str or 'по слово' in lem_str or 'по данный' in lem_str or 'по мнение' in lem_str or 'по оценка' in lem_str:
        result_dict['attribution'] = 1
    if 'в момент' in lem_str or 'на момент' in lem_str or 'на тот момент' in lem_str or 'в этот момент' in lem_str or 'в рамка' in lem_str or 'в условие' in lem_str or 'на этот фон' in lem_str or 'при этот' in lem_str or 'при это' in lem_str:
        result_dict['background'] = 1
    if 'в итог' in lem_str or 'в настоящий время' in lem_str:
        result_dict['sequence'] = 1
    if 'кроме тот' in lem_str or 'к тот же' in lem_str:
        result_dict['joint'] = 1
    if 'до такой степень' in lem_str:
        result_dict['cause-effect'] = 1
    if 'с цель' in lem_str:
        result_dict['purpose'] = 1
    if 'все же' in lem_str or 'всё же' in lem_str or 'весь же' in lem_str or 'с другой сторона' in lem_str or 'в отличие' in lem_str or 'в то время как' in lem_str or 'а не' in lem_str:
        result_dict['contrast'] = 1
    if 'в то время как' in lem_str:
        result_dict['comparison'] = 1
    if ('играть' in lem_str or 'сыграть' in lem_str) and 'роль' in lem_str :
        result_dict['interpretation-evaluation'] = 1
    if 'в отличие' in lem_str:
        result_dict['concession'] = 1
    return result_dict


def detect_head(parsed):
    raw_conll = parsed.to_conll(3)
    conll = [elem.split('\t') for elem in raw_conll.split('\n')]
    for i,row in enumerate(conll):
        if row[2] == '0':
            return i


def generate_feature_matrix(pairs):
    c_vect = CountVectorizer(min_df=5, ngram_range=(1,3), tokenizer=word_tokenize)
    pos_vect = CountVectorizer(tokenizer=iterate_pos, ngram_range=(1,3), lowercase=False)
    mp = MaltParser("/home/lena/opt/maltparser-1.9.2","/home/lena/opt/russian.mco")
    model_d = Doc2Vec.load('vec/model_d.w2v')
    model_w = Word2Vec.load('vec/model_w.w2v')
    DataDict = {'edu1_position': [],
                'edu2_position': [],
                'edu1_endsent': [],
                'edu1_startsent': [],
                'edu2_endsent': [],
                'edu2_startsent': [],
                'edu1_len': [],
                'edu2_len': [],
                'same_tokens': [],
                'distance': [],
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
        DataDict['distance'].append(int(pair.edu2.position)-int(pair.edu1.position)-1)
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

    # # СИНТАКСИС Word2Vec - векторы head ЭДЕ, POS-теги head ЭДЕ
    # head_ids_edu1 = [detect_head(mp.parse_one(pair.edu1.tokens)) for pair in pairs]
    # head_ids_edu2 = [detect_head(mp.parse_one(pair.edu2.tokens)) for pair in pairs]
    # head_vectors_edu1 = csr_matrix(np.array([w2v_word_vector(model_w, pairs[i].edu1.tokens[head_ids_edu1[i]]) for i in range(len(pairs))]))
    # head_vectors_edu2 = csr_matrix(np.array([w2v_word_vector(model_w, pairs[i].edu2.tokens[head_ids_edu2[i]]) for i in range(len(pairs))]))
    # head_pos_edu1 = pos_vect.transform([[pairs[i].edu1.pos[head_ids_edu1[i]]] for i in range(len(pairs))])
    # head_pos_edu2 = pos_vect.transform([[pairs[i].edu2.pos[head_ids_edu2[i]]] for i in range(len(pairs))])

    X_sparse = csr_matrix(np.array(X))
    X_concat = hstack((X_sparse, edus1_vect, edus2_vect, pos1_vect, pos2_vect, d2v1, d2v2,
                       w2v1_first, w2v2_first, w2v1_last, w2v2_last))
    print(X_concat.shape)
    return X_concat

def get_feature_names(pairs):
    feature_names = []
    c_vect = CountVectorizer(min_df=5, ngram_range=(1,3), tokenizer=word_tokenize)
    pos_vect = CountVectorizer(tokenizer=iterate_pos, ngram_range=(1,3), lowercase=False)
    mp = MaltParser("/home/lena/opt/maltparser-1.9.2","/home/lena/opt/russian.mco")
    model_d = Doc2Vec.load('vec/model_d.w2v')
    model_w = Word2Vec.load('vec/model_w.w2v')
    DataDict = {'edu1_position': [],
                'edu2_position': [],
                'edu1_endsent': [],
                'edu1_startsent': [],
                'edu2_endsent': [],
                'edu2_startsent': [],
                'edu1_len': [],
                'edu2_len': [],
                'same_tokens': [],
                'distance': [],
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
        DataDict['distance'].append(int(pair.edu2.position)-int(pair.edu1.position)-1)
        for rel_name in ['attribution','cause-effect','concession','condition','contrast','elaboration','joint','purpose']:
            DataDict[rel_name+'1'].append(markers_dict1[rel_name])
            DataDict[rel_name+'2'].append(markers_dict2[rel_name])


    X = pd.DataFrame(DataDict)
    feature_names.extend(X.columns)
    # векторайзер по словам
    all_texts = [pair.edu1.text for pair in pairs] + [pair.edu2.text for pair in pairs]
    c_vect.fit(all_texts)
    # edus1_vect = c_vect.transform([pair.edu1.text for pair in pairs])
    # edus2_vect = c_vect.transform([pair.edu2.text for pair in pairs])

    feature_names.extend([i+'1' for i in c_vect.get_feature_names() ])
    feature_names.extend([i+'2' for i in c_vect.get_feature_names() ])

    # векторайзер по тегам частей речи
    all_pos = [pair.edu1.pos for pair in pairs] + [pair.edu2.pos for pair in pairs]
    pos_vect.fit(all_pos)
    pos1_vect = pos_vect.transform([pair.edu1.pos for pair in pairs])
    pos2_vect = pos_vect.transform([pair.edu2.pos for pair in pairs])

    feature_names.extend([i+'1' for i in pos_vect.get_feature_names() ])
    feature_names.extend([i+'2' for i in pos_vect.get_feature_names() ])

    # Doc2Vec - вектор ЭДЕ
    d2v1 = csr_matrix(np.array([model_d.infer_vector(pair.edu1.tokens) for pair in pairs]))
    d2v2 = csr_matrix(np.array([model_d.infer_vector(pair.edu1.tokens) for pair in pairs]))

    feature_names.extend(['d2v1'+str(i) for i in range(100)])
    feature_names.extend(['d2v2'+str(i) for i in range(100)])

    # Word2Vec - векторы первого и последнего токена в ЭДЕ
    w2v1_first = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu1.tokens[0]) for pair in pairs]))
    w2v2_first = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu2.tokens[0]) for pair in pairs]))
    w2v1_last = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu1.tokens[-1]) for pair in pairs]))
    w2v2_last = csr_matrix(np.array([w2v_word_vector(model_w, pair.edu2.tokens[-1]) for pair in pairs]))

    feature_names.extend(['w2v1_first'+str(i) for i in range(100)])
    feature_names.extend(['w2v2_first'+str(i) for i in range(100)])

    feature_names.extend(['w2v1_last'+str(i) for i in range(100)])
    feature_names.extend(['w2v2_last'+str(i) for i in range(100)])

    # # СИНТАКСИС Word2Vec - векторы head ЭДЕ, POS-теги head ЭДЕ
    # head_ids_edu1 = [detect_head(mp.parse_one(pair.edu1.tokens)) for pair in pairs]
    # head_ids_edu2 = [detect_head(mp.parse_one(pair.edu2.tokens)) for pair in pairs]
    # head_vectors_edu1 = csr_matrix(np.array([w2v_word_vector(model_w, pairs[i].edu1.tokens[head_ids_edu1[i]]) for i in range(len(pairs))]))
    # head_vectors_edu2 = csr_matrix(np.array([w2v_word_vector(model_w, pairs[i].edu2.tokens[head_ids_edu2[i]]) for i in range(len(pairs))]))
    # head_pos_edu1 = pos_vect.transform([[pairs[i].edu1.pos[head_ids_edu1[i]]] for i in range(len(pairs))])
    # head_pos_edu2 = pos_vect.transform([[pairs[i].edu2.pos[head_ids_edu2[i]]] for i in range(len(pairs))])

    # X_sparse = csr_matrix(np.array(X))
    # X_concat = hstack((X_sparse, edus1_vect, edus2_vect, pos1_vect, pos2_vect, d2v1, d2v2,
    #                    w2v1_first, w2v2_first, w2v1_last, w2v2_last))
    print(len(feature_names))

    return feature_names

# def generate_feature_matrix_(pairs):
#     c_vect = CountVectorizer(min_df=5, tokenizer=word_tokenize)
#     pos_vect = CountVectorizer(tokenizer=iterate_pos, lowercase=False)
#     DataDict = {'edu1_position': [],
#                 'edu2_position': [],
#                 'edu1_endsent': [],
#                 'edu1_startsent': [],
#                 'edu2_endsent': [],
#                 'edu2_startsent': [],
#                 'edu1_len': [],
#                 'edu2_len': [],
#                 'same_tokens': []}
#     for pair in pairs:
#         DataDict['edu1_position'].append(int(pair.edu1.position))
#         DataDict['edu2_position'].append(int(pair.edu2.position))
#         DataDict['edu1_endsent'].append(int(pair.edu1.sentence_end))
#         DataDict['edu2_endsent'].append(int(pair.edu2.sentence_end))
#         DataDict['edu1_startsent'].append(int(pair.edu1.sentence_start))
#         DataDict['edu2_startsent'].append(int(pair.edu2.sentence_start))
#         DataDict['edu1_len'].append(len(pair.edu1.tokens))
#         DataDict['edu2_len'].append(len(pair.edu2.tokens))
#         # количество совпадающих токенов (леммы)
#         DataDict['same_tokens'].append(len(set(pair.edu1.lemmatized_tokens).intersection(
#             pair.edu2.lemmatized_tokens)))
#
#     X = pd.DataFrame(DataDict)
#
#     # векторайзер по словам
#     all_texts = [pair.edu1.text for pair in pairs] + [pair.edu2.text for pair in pairs]
#     c_vect.fit(all_texts)
#     edus1_vect = c_vect.transform([pair.edu1.text for pair in pairs])
#     edus2_vect = c_vect.transform([pair.edu2.text for pair in pairs])
#
#     # векторайзер по тегам частей речи
#     all_pos = [pair.edu1.pos for pair in pairs] + [pair.edu2.pos for pair in pairs]
#     pos_vect.fit(all_pos)
#     pos1_vect = pos_vect.transform([pair.edu1.pos for pair in pairs])
#     pos2_vect = pos_vect.transform([pair.edu2.pos for pair in pairs])
#
#     X_sparse = csr_matrix(np.array(X))
#     X_concat = hstack((X_sparse, edus1_vect, edus2_vect, pos1_vect, pos2_vect))
#     print(X_concat.shape)
#     return X_concat

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
