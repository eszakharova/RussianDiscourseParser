import pandas as pd
import numpy as np
import pickle
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import classification_report
import regex as re
from collections import Counter
from imblearn.over_sampling import SMOTE


def iterate_pos(pos_list):
    for pos in pos_list:
        yield pos


def rel_type_only(y):
    return [re.sub('_(SN|NS|M)', '', elem) for elem in y]


def nucl_only(y):
    return [elem.split('_')[1] if elem != 'no_relation' else elem for elem in y ]


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


def generate_feature_matrix(pairs):
    c_vect = CountVectorizer(min_df=5, tokenizer=word_tokenize)
    pos_vect = CountVectorizer(tokenizer=iterate_pos, lowercase=False)
    DataDict = {'edu1_position': [],
                'edu2_position': [],
                'edu1_endsent': [],
                'edu1_startsent': [],
                'edu2_endsent': [],
                'edu2_startsent': [],
                'edu1_len': [],
                'edu2_len': []}
    for pair in pairs:
        DataDict['edu1_position'].append(int(pair.edu1.position))
        DataDict['edu2_position'].append(int(pair.edu2.position))
        DataDict['edu1_endsent'].append(int(pair.edu1.sentence_end))
        DataDict['edu2_endsent'].append(int(pair.edu2.sentence_end))
        DataDict['edu1_startsent'].append(int(pair.edu1.sentence_start))
        DataDict['edu2_startsent'].append(int(pair.edu2.sentence_start))
        DataDict['edu1_len'].append(len(pair.edu1.tokens))
        DataDict['edu2_len'].append(len(pair.edu2.tokens))
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

def smote_oversampling(X, y):
	sm = SMOTE(random_state=669, k_neighbors=1)
	X_res, y_res = sm.fit_sample(X, y)
	print(X_res.shape)
	return X_res, y_res
