import os
import time
import tensorflow as tf

from duplicate_question_pairs.utilities.history_logger import HistoryLogger
from duplicate_question_pairs.utilities.preprocessor import preprocess


def find_latest(model_name):
    dirs = [int(o) for o in os.listdir(model_name) if os.path.isdir(os.path.join(model_name, o))]
    latest_dir = max(dirs)
    return os.path.join(model_name, str(latest_dir), str(latest_dir))


def main():
    file_path = "data/train_small.csv"
    X_test, Y_test, embed_construct = preprocess(file_path, is_train=False)
    start = time.time()
    model_name = os.path.join(os.getcwd(), 'model_name')
    model_name = find_latest(model_name)
    test_path = os.path.join(os.getcwd(), 'data/test_data.txt')
    logger = HistoryLogger.load(model_name)
    best_model = logger.best_model
    graph = tf.Graph()
    config = logger.config
    data_gen = get_batch_data_iterator(n_epoch=1, data=(X_test, Y_test), seq_length=config.maxSeqLength,
                                       batch_size=1, mode='train')
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            graph_path = best_model[:best_model.rfind('_')] + '_' + str(config.evaluate_every)
            saver = tf.train.import_meta_graph("{}.meta".format(graph_path))
            saver.restore(sess, logger.best_model)


if __name__ == '__main__':
    main()
