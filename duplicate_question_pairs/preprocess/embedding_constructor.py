import yaml
import numpy as np
import os
import pickle
from duplicate_question_pairs.definitions import ROOT_DIR


"""
embed path : embeddings file path. eg. glove vectors
embedding matrix path : path of the embedding matrix constructed
embedding: embedding loaded from embed_path file. e.g. glove vectors
embedding matrix: embedding matrix constructed from embeddings
"""


class EmbeddingConstructor:
    def __init__(self):
        self.root_path = ROOT_DIR
        self.paths = self._get_paths()
        self.embed_path = self.paths['embed_path']
        self.embed_size = int(self.paths['embed_size'])
        self.word2Id_path = self._get_abs_path(self.paths['word2Id_path']) + '_' + str(self.embed_size)
        self.id2Word_path = self._get_abs_path(self.paths['id2Word_path']) + '_' + str(self.embed_size)
        self.embed_matrix_path = self._get_abs_path(self.paths['embed_matrix_path']) + '_' + str(self.embed_size)
        self.word2Id = None
        self.id2Word = None
        self.embed_matrix = None
        self.unknown = 'unk'
        self.unknown_vector = np.random.random(self.embed_size)
        self.pad = 'pad'
        self.pad_vector = np.zeros(self.embed_size)
        self.embedding = None

    def construct(self):
        if self._vocab_dicts_and_matrix_created():
            self._load_vocab_and_matrix()
        else:
            self.embedding = self._load_embedding()
            self._build_and_save_vocab_and_matrix()
        return self

    def _get_paths(self):
        with open(os.path.join(self.root_path, "resources/paths.yaml"), "r+", encoding='utf-8') as stream:
            try:
                return yaml.load(stream)
            except yaml.YAMLError as exc:
                raise RuntimeWarning("glove vectors not found or wrong path provided.", exc)

    def _get_abs_path(self, path):
        return os.path.join(self.root_path, path)

    def _load_embedding(self):
        embedding = {}
        with open(self.embed_path, 'r+', encoding='utf-8') as embed_file:
            for line in embed_file:
                split_line = line.split()
                embed = np.array([float(val) for val in split_line[-self.embed_size: len(split_line)]])
                word = ""
                for elem in split_line[:-self.embed_size]:
                    word += elem
                embedding[word] = embed
        return embedding

    def _vocab_dicts_and_matrix_created(self):
        return os.path.isfile(self.word2Id_path) and os.path.isfile(self.id2Word_path) and \
               os.path.isfile(self.embed_matrix_path)

    def _load_vocab_and_matrix(self):
        word2Id_file = open(self.word2Id_path, "rb")
        self.word2Id = pickle.load(word2Id_file)
        word2Id_file.close()

        id2Word_file = open(self.id2Word_path, "rb")
        self.id2Word = pickle.load(id2Word_file)
        id2Word_file.close()

        embedding_matrix_file = open(self.embed_matrix_path, "rb")
        self.embed_matrix = pickle.load(embedding_matrix_file)
        embedding_matrix_file.close()

        assert (self.word2Id is not None and self.id2Word is not None and self.embed_matrix is not  None)

    def _build_and_save_vocab_and_matrix(self):
        self.embed_matrix = np.zeros((len(self.embedding.keys()) + 2, self.embed_size))
        self.word2Id = dict()
        self.id2Word = dict()
        self.word2Id[self.pad] = 0
        self.id2Word[0] = self.pad
        self.embed_matrix[0] = self.pad_vector

        for word_id, word in enumerate(self.embedding.keys()):
            self.word2Id[word] = word_id + 1
            self.id2Word[word_id + 1] = word
            self.embed_matrix[word_id + 1] = self.embedding[word]

        self.word2Id[self.unknown] = len(self.embed_matrix) - 1
        self.id2Word[len(self.embed_matrix) - 1] = self.unknown
        self.embed_matrix[len(self.embed_matrix) - 1] = self.unknown_vector
        self._save_vocab_and_matrix()

    def _save_vocab_and_matrix(self):
        word2_id_file = open(self.word2Id_path, "wb")
        pickle.dump(self.word2Id, word2_id_file)
        word2_id_file.close()

        id2_word_file = open(self.id2Word_path, "wb")
        pickle.dump(self.id2Word, id2_word_file)
        id2_word_file.close()

        embedding_matrix_file = open(self.embed_matrix_path, "wb")
        pickle.dump(self.embed_matrix, embedding_matrix_file)
        embedding_matrix_file.close()

def test():
    embedding_constructor = EmbeddingConstructor()
    embedding_constructor.construct()


if __name__ == '__main__':
    test()
