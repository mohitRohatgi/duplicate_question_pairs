from duplicate_question_pairs.model.model import Model
from duplicate_question_pairs.preprocess.preprocessor import preprocess

import tensorflow as tf


def main():
    file_path = "data/train_small.csv"
    X_train, X_valid, Y_train, Y_valid, embed_construct = preprocess(file_path)

    model = Model(len(embed_construct.embed_matrix))
    with tf.Graph().as_default() as graph:
        with tf.Session(graph=graph) as sess:
            model.initialise(sess, embed_construct.embed_matrix)


if __name__ == '__main__':
    main()
