import os
import time
import tensorflow as tf

from config.config import Config
from model.model import Model
from utilities.history_logger import HistoryLogger
from utilities.preprocessor import preprocess
from utilities.utils import get_batch_data_iterator


def main():
    file_path = "data/train_small.csv"
    X_train, X_valid, Y_train, Y_valid, embed_construct = preprocess(file_path)
    save_meta_graph = True
    config = Config()
    logger = HistoryLogger(config)
    data_gen = get_batch_data_iterator(n_epoch=config.n_epoch, data=(X_train, X_valid, Y_train, Y_valid),
                                       seq_length=config.maxSeqLength, batch_size=config.batchSize, mode='train')
    with tf.Graph().as_default() as graph:
        model = Model(len(embed_construct.embed_matrix))
        with tf.Session(graph=graph) as sess:
            model.initialise(sess, embed_construct.embed_matrix)
            step = 0
            model_name = os.path.join(os.getcwd(), 'model')
            model_no = int(time.time())
            model_name = os.path.join(model_name, str(model_no))
            logger_path = os.path.join(model_name, str(model_no))
            if not os.path.exists(model_name):
                os.makedirs(model_name)
            saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1000000)
            train_batch_data, train_label_batch, valid_batch_data, valid_batch_label = data_gen.__next__()
            mean_train_loss = 0.0
            mean_train_accuracy = 0.0
            valid_data = []
            valid_label = []
            while train_batch_data is not None:
                train_loss, train_accuracy, train_prediction = model.run_batch(sess, train_batch_data, True,
                                                                               train_label_batch)
                valid_data.append(valid_batch_data)
                valid_label.append(valid_batch_label)
                step += 1
                mean_train_loss += train_loss
                mean_train_accuracy += train_accuracy
                if step % config.evaluate_every == 0:
                    mean_valid_loss, mean_valid_accuracy = 0.0, 0.0

                    for i in range(config.evaluate_every):
                        valid_loss, valid_accuracy, valid_prediction = model.run_batch(sess, valid_data[i], False,
                                                                                       valid_label[i])
                        mean_valid_loss += valid_loss
                        mean_valid_accuracy += valid_accuracy
                        saver.save(sess, os.path.join(logger_path + '_' + str(step)), write_meta_graph=save_meta_graph)
                        save_meta_graph = False

                    mean_valid_loss /= config.evaluate_every
                    mean_valid_accuracy /= config.evaluate_every
                    mean_train_loss /= config.evaluate_every
                    mean_train_accuracy /= config.evaluate_every
                    logger.add(mean_train_loss, mean_train_accuracy, mean_valid_loss, mean_valid_accuracy, step)
                    logger.save(logger_path)
                    print("step = ", step, "mean valid loss = ", mean_valid_loss, " mean valid accuracy = ",
                          mean_valid_accuracy, " config = ", config)
                    print("step = ", step, "mean train loss = ", mean_train_loss, " mean train accuracy = ",
                          mean_train_accuracy, " config = ", config)
                    valid_data = []
                    valid_label = []
                    mean_train_loss = 0.0
                    mean_train_accuracy = 0.0
                train_batch_data, train_label_batch, valid_batch_data, valid_batch_label = data_gen.__next__()


if __name__ == '__main__':
    main()
