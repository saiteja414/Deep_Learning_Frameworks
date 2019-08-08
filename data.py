import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train_data = self.tokenize(os.path.join(path, 'train_flicker.txt'))
        self.val_data  = self.tokenize(os.path.join(path, 'val_flicker.txt'))
        self.test_data = self.tokenize(os.path.join(path, 'test_flicker.txt'))
        self.word2idx = self.dictionary.word2idx
        self.idx2word = self.dictionary.idx2word

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            data = []
            # self.dictionary.add_word("#START#")
            for line in f:
                words = line.split()
                words = words[1:]
                # tup = ()
                inp = ['#START#'] + words[:-1]
                targets = words
                tup = tuple([inp,targets])
                data.append(tup)
                for word in words:
                    self.dictionary.add_word(word)
        return data