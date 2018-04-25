import re
import numpy as np
from nltk.stem.porter import PorterStemmer


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
        sent_vectors.append(indices)
    return sent_vectors


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

def batch_iter(nepochs, X, Y=None):
    pass
