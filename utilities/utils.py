import re
import numpy as np

from nltk.stem.porter import PorterStemmer
from utilities.text_representer import TextRepresenter


def clean_question(question):
    question1_cleaned = clean_text(question.lower())
    question1_words = question1_cleaned.split()
    ps = PorterStemmer()
    return [ps.stem(word) for word in question1_words]


def index_text_to_word_id(sentences, word2_id, pad='pad', max_seq_length=20):
    sent_vectors = []
    for j, sent in enumerate(sentences):
        indices = np.ones(max_seq_length) * word2_id[pad]
        for i, word in enumerate(sent):
            if i >= max_seq_length:
                continue
            if word in word2_id:
                indices[i] = word2_id[word]
            else:
                indices[i] = word2_id[pad]
        sent_vectors.append(TextRepresenter(sent, indices))
    return np.array(sent_vectors)


def clean_text(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def get_batch_data(X, Y, batch_size):
    indices = np.random.randint(0, len(X), batch_size)
    question1_batch = np.array([x.values for x in X['question1'].values[indices]])
    question2_batch = np.array([x.values for x in X['question2'].values[indices]])
    qid1_batch = X['qid1'].values[indices]
    qid2_batch = X['qid2'].values[indices]
    train_label_batch = Y.values[indices]
    return (question1_batch, question2_batch, qid1_batch, qid2_batch), train_label_batch


def get_batch_data_iterator(n_epoch, data, batch_size, mode='train'):
    if mode == 'train':
        X_train, X_valid, Y_train, Y_valid = data
        num_batches_per_epoch = int((len(X_train)) / batch_size) + 1
        for i in range(n_epoch):
            for j in range(num_batches_per_epoch):
                train_batch_data, train_label_batch = get_batch_data(X=X_train, Y=Y_train, batch_size=batch_size)
                valid_batch_data, valid_label_batch = get_batch_data(X=X_valid, Y=Y_valid, batch_size=batch_size)
                yield train_batch_data, train_label_batch, valid_batch_data, valid_label_batch
    else:
        X_test, Y_test = data
