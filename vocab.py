import pickle
import argparse
from collections import Counter
from util import *
import os
from sentencepiece import generate_encoding_model


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
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(raw_data, threshold):
    src_counter = Counter()
    tgt_counter = Counter()

    for i, line in enumerate(raw_data.src):
        tokens = line.split()
        src_counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized in the source file." % (i, len(raw_data.src)))

    src_words = [word for word, cnt in src_counter.items() if cnt >= threshold]

    src_vocab = Vocabulary()
    src_vocab.add_word('<pad>')
    src_vocab.add_word('<start>')
    src_vocab.add_word('<end>')
    src_vocab.add_word('<unk>')
    for i, word in enumerate(src_words):
        src_vocab.add_word(word)

    for i, line in enumerate(raw_data.tgt):
        tokens = line.split()
        tgt_counter.update(tokens)

        if i % 1000 == 0:
            print("[%d/%d] Tokenized in the target file." % (i, len(raw_data.tgt)))

    tgt_words = [word for word, cnt in tgt_counter.items() if cnt >= threshold]

    tgt_vocab = Vocabulary()
    tgt_vocab.add_word('<pad>')
    tgt_vocab.add_word('<start>')
    tgt_vocab.add_word('<end>')
    tgt_vocab.add_word('<unk>')
    for i, word in enumerate(tgt_words):
        tgt_vocab.add_word(word)

    print(src_vocab.word2idx.keys())
    return src_vocab, tgt_vocab

def main(args):
    raw_data = RawDataset(args.data_path, types=['train', 'val', 'test'],
                          src_language=args.src_language, tgt_language=args.tgt_language)
    bpe_data = BPEDataset(args.data_path, types=['train', 'val', 'test'],
                          src_language=args.src_language, tgt_language=args.tgt_language)
    src_vocab, tgt_vocab = build_vocab(raw_data, args.threshold)
    src_bpe_vocab, tgt_bpe_vocab = build_vocab(bpe_data, 0)
    src_vocab_path = args.src_vocab_path
    tgt_vocab_path = args.tgt_vocab_path
    src_encoding_path = args.src_encoding_path
    tgt_encoding_path = args.tgt_encoding_path
    if not os.path.exists('models/'):
        os.makedirs('models/')
    with open(src_vocab_path, 'wb') as f:
        pickle.dump(src_vocab, f)
    with open(tgt_vocab_path, 'wb') as f:
        pickle.dump(tgt_vocab, f)
    with open(src_encoding_path_path, 'wb') as f:
        pickle.dump(src_bpe_vocab, f)
    with open(tgt_encoding_path, 'wb') as f:
        pickle.dump(tgt_bpe_vocab, f)


    print('Total source vocabulary size: %d' % len(src_vocab))
    print("Saved the source vocabulary wrapper to '%s'" % src_vocab_path)
    print('Total target vocabulary size: %d' % len(tgt_vocab))
    print("Saved the target vocabulary wrapper to '%s'" % tgt_vocab_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='dataset/data/task1/tok/')
    parser.add_argument('--src_vocab_path', type=str, default='./models/src_vocab.pkl')
    parser.add_argument('--image_feature_dir', type=str, default='../features_resnet50/')
    parser.add_argument('--tgt_vocab_path', type=str, default='./models/tgt_vocab.pkl')
    parser.add_argument('--src_encoding_path', type=str, default='./models/src_bpe_vocab.pkl')
    parser.add_argument('--tgt_encoding_path', type=str, default='./models/tgt_bpe_vocab.pkl')
    parser.add_argument('--src_vocab_size', type=int, default=4000)
    parser.add_argument('--tgt_vocab_size', type=int, default=6000)
    parser.add_argument('--threshold', type=int, default=3)
    parser.add_argument('--src_language', type=str, default='en')
    parser.add_argument('--tgt_language', type=str, default='de')
    args = parser.parse_args()
    main(args)
