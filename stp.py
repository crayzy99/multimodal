import os
import errno
import sentencepiece as spm
import re
import logging
import argparse

logging.basicConfig(level=logging.INFO)

def generate_encoding_model(language, vocab_size):
    data_path = 'dataset/data/task1/tok/'
    train = data_path + 'train.lc.norm.tok.' + language
    val   = data_path + 'val.lc.norm.tok.' + language
    test  = data_path + 'test_2016_flickr.lc.norm.tok.' + language

    print("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip() for line in open(x, mode='r').read().split("\n")]
    prepro_train = _prepro(train)
    prepro_eval = _prepro(val)
    prepro_test = _prepro(test)

    print("# write preprocessed files to disk")
    def _write(sents, fname):
        with open(fname, mode='w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train+prepro_eval+prepro_test, data_path + 'total.'+language)

    print("# Train a BPE model of language {} with sentencepiece".format(language))
    train = '--input=dataset/data/task1/tok/total.{} --pad_id=0 --unk_id=3 \
             --bos_id=1 --eos_id=2\
             --model_prefix=models/{}_bpe --vocab_size={} \
             --model_type=bpe'.format(language, language, vocab_size)
    spm.SentencePieceTrainer.Train(train)

    print("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("models/"+language+"_bpe.model")

    print("# Segment")
    def _segment_and_write(sents, fname):
        with open(fname,mode= "w") as fout:
            for sent in sents:
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train, data_path + "train." + language + ".bpe")
    _segment_and_write(prepro_eval, data_path + "val." + language + ".bpe")
    _segment_and_write(prepro_test, data_path + "test." + language + ".bpe")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='en')
    parser.add_argument('--vocab_size', type=int, default=4000)
    args = parser.parse_args()
    generate_encoding_model(args.language, args.vocab_size)
    print("Done")
