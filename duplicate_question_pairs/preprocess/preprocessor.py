import os
import pandas as pd

from duplicate_question_pairs.preprocess.embedding_constructor import EmbeddingConstructor
from duplicate_question_pairs.preprocess.utils import clean_question, index_text_to_word_id
from sklearn.model_selection import train_test_split


def main():
    filename = "data/train_small.csv"
    filename = os.path.abspath(os.path.join(os.pardir, os.pardir, filename))
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
    return X_train, X_valid, X_test, Y_train, Y_valid, Y_test, embed_construct


if __name__ == '__main__':
    main()
