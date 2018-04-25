import os
import pandas as pd
import pickle

from duplicate_question_pairs.preprocess.embedding_constructor import EmbeddingConstructor
from duplicate_question_pairs.preprocess.utils import clean_question, index_text_to_word_id
from sklearn.model_selection import train_test_split


def save_file(file_name, file_object):
    file = open(os.path.join('resources', file_name + '.pkl'), "wb")
    pickle.dump(file_object, file, pickle.HIGHEST_PROTOCOL)
    file.close()


def load_files(is_train):
    if is_train:
        filenames = ['X_train', 'X_valid', 'Y_train', 'Y_valid', 'embed_construct']
    else:
        filenames = ['X_test', 'Y_test', 'embed_construct']
    files = []
    for filename in filenames:
        with open(os.path.join('resources', filename + '.pkl'), "rb") as file:
            files.append(pickle.load(file))
    return files


def data_is_saved(is_train):
    if is_train:
        return check_if_files_created(files=['X_train', 'X_valid', 'Y_train', 'Y_valid', 'embed_construct'])
    return check_if_files_created(files=['X_test', 'Y_test', 'embed_construct'])


def check_if_files_created(files):
    for file in files:
        if os.path.isfile(os.path.join('resources/', file + '.pkl')):
            continue
        else:
            return False
    return True


def preprocess(file_path, is_train=True):
    if data_is_saved(is_train):
        return load_files(is_train)
    filename = os.path.abspath(os.path.join(os.pardir, file_path))
    data = pd.read_csv(filename)
    embed_construct = EmbeddingConstructor().construct()
    Y = data.is_duplicate
    X = data.drop(columns=['is_duplicate', 'id'])
    X['question1'] = X['question1'].apply(lambda x: clean_question(x))
    X['question2'] = X['question2'].apply(lambda x: clean_question(x))
    X['question1'] = index_text_to_word_id(X['question1'], embed_construct.word2Id, embed_construct.pad)
    X['question2'] = index_text_to_word_id(X['question2'], embed_construct.word2Id, embed_construct.pad)
    Y = Y.apply(lambda x: int(x))
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    del X, Y
    save_file('X_train', X_train)
    save_file('X_valid', X_valid)
    save_file('X_test', X_test)
    save_file('Y_train', X_train)
    save_file('Y_valid', X_valid)
    save_file('Y_test', X_test)
    save_file('embed_construct', embed_construct)
    if is_train:
        return X_train, X_valid, Y_train, Y_valid, embed_construct
    return X_test, Y_test, embed_construct


if __name__ == '__main__':
    preprocess()
