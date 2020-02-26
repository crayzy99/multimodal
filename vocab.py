import pickle
import argparse
from collections import Counter
from util import *


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[b'<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(data_path, threshold, src_language, tgt_language):
    raw_data = RawDataset(data_path, types=['train', 'val', 'test'],
                          src_language=src_language, tgt_language=tgt_language)
    src_counter = Counter()
    tgt_counter = Counter()

    for i, line in enumerate(en_de.src):
        tokens = line.split()
        src_counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized in the source file." % (i, len(en_de.src)))

    src_words = [word for word, cnt in src_counter.items() if cnt >= threshold]

    src_vocab = Vocabulary()
    src_vocab.add_word(b'<pad>')
    src_vocab.add_word(b'<start>')
    src_vocab.add_word(b'<end>')
    src_vocab.add_word(b'<unk>')
    for i, word in enumerate(src_words):
        src_vocab.add_word(word)

    for i, line in enumerate(en_de.tgt):
        tokens = line.split()
        tgt_counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized in the target file." % (i, len(en_de.tgt)))

    tgt_words = [word for word, cnt in tgt_counter.items() if cnt >= threshold]

    tgt_vocab = Vocabulary()
    tgt_vocab.add_word(b'<pad>')
    tgt_vocab.add_word(b'<start>')
    tgt_vocab.add_word(b'<end>')
    tgt_vocab.add_word(b'<unk>')
    for i, word in enumerate(tgt_words):
        tgt_vocab.add_word(word)

    print(tgt_vocab.word2idx.keys())
    return src_vocab, tgt_vocab


def main(args):
    src_vocab, tgt_vocab = build_vocab(args.data_path, args.threshold, args.src_language, args.tgt_language)
    src_vocab_path = args.src_vocab_path
    tgt_vocab_path = args.tgt_vocab_path
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(tgt_vocab_path, 'wb') as f:
        pickle.dump(tgt_vocab, f)

    print('Total source vocabulary size: %d' % len(src_vocab))
    print("Saved the source vocabulary wrapper to '%s'" % src_vocab_path)
    print('Total target vocabulary size: %d' % len(tgt_vocab))
    print("Saved the target vocabulary wrapper to '%s'" % tgt_vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../task1/tok/')
    parser.add_argument('--src_vocab_path', type=str, default='./models/src_vocab.pkl')
    parser.add_argument('--image_feature_dir', type=str, default='../features_resnet50/')
    parser.add_argument('--tgt_vocab_path', type=str, default='./models/tgt_vocab.pkl')
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--src_language', type=str, default='en')
    parser.add_argument('--tgt_language', type=str, default='de')
    args = parser.parse_args()
    main(args)