import os
import time
import tensorflow as tf

from utilities.history_logger import HistoryLogger
from utilities.preprocessor import preprocess
from utilities.utils import get_batch_data_iterator


def find_latest(model_name):
    dirs = [int(o) for o in os.listdir(model_name) if os.path.isdir(os.path.join(model_name, o))]
    latest_dir = max(dirs)
    return os.path.join(model_name, str(latest_dir), str(latest_dir))


def main():
    file_path = "data/train.csv"
    start = time.time()
    model_name = os.path.join(os.getcwd(), 'saved_model')
    model_name = find_latest(model_name)
    logger = HistoryLogger.load(model_name)
    best_model = logger.best_model
    graph = tf.Graph()
    config = logger.config
    X_test, Y_test, embed_construct = preprocess(file_path, is_train=False, max_seq_length=config.maxSeqLength)
    data_gen = get_batch_data_iterator(n_epoch=1, data=(X_test, Y_test), batch_size=config.batchSize, mode='test',
                                       is_label=True)
    with graph.as_default():
        sess = tf.Session(graph=graph)
        with sess.as_default():
            graph_path = best_model[:best_model.rfind('_')] + '_' + str(config.evaluate_every)
            saver = tf.train.import_meta_graph("{}.meta".format(graph_path))
            saver.restore(sess, logger.best_model)

            # Get the placeholders from the graph by name
            embedding_placeholder = graph.get_operation_by_name('embed_matrix').outputs[0]
            word_embedding = graph.get_operation_by_name('embeddings/embeddings').outputs[0]
            embed_init = tf.assign(word_embedding, embedding_placeholder)
            # embed_init = word_embedding.assign(embedding_placeholder)
            input_placeholder_q1 = graph.get_operation_by_name('q1').outputs[0]
            input_placeholder_q2 = graph.get_operation_by_name('q2').outputs[0]
            qid1 = graph.get_operation_by_name('qid1').outputs[0]
            qid2 = graph.get_operation_by_name('qid2').outputs[0]
            label = graph.get_operation_by_name('label').outputs[0]
            dropout_keep_prob = graph.get_operation_by_name('dropout_keep').outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name('linear/prediction').outputs[0]
            accuracy = graph.get_operation_by_name('linear/accuracy').outputs[0]

            sess.run([embed_init], {embedding_placeholder: embed_construct.embed_matrix})

            # Collect the predictions here
            all_predictions = []
            total_correct = 0
            total = 0

            while True:
                try:
                    (batch_q1, batch_q2, batch_qid1, batch_qid2), label_batch = data_gen.__next__()
                    feed_dict = {
                        dropout_keep_prob: 1.0,
                        input_placeholder_q1: batch_q1,
                        input_placeholder_q2: batch_q2,
                        qid1: batch_qid1,
                        qid2: batch_qid2,
                    }
                    if label_batch is not None:
                        feed_dict[label] = label_batch
                        batch_predictions, batch_accuracy = sess.run([predictions, accuracy], feed_dict)
                        total_correct += int(round(batch_accuracy * config.batchSize))
                        total += config.batchSize
                        print("total correct = ", str(total_correct), " out of total = ", str(total))
                    else:
                        batch_predictions = sess.run(predictions, feed_dict=feed_dict)
                    all_predictions.extend(batch_predictions)
                except:
                    break

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'testPrediction.bin'), 'w') as f:
            f.write("id,label\n")
            for line_id, prediction in enumerate(all_predictions):
                f.write(str(line_id) + "," + str(prediction) + "\n")
        print("time taken = ", time.time() - start)


if __name__ == '__main__':
    main()
