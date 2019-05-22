import argparse
import torch
import torch.nn as nn
import numpy as np
import argparse
import pickle
import random
import os
from data_loader import get_loader
from torch.autograd import Variable
from vocab import Vocabulary
from model import Encoder, AttnDecoder_1


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='../task1/tok/')
parser.add_argument('--image_feature_dir', type=str, default='../features_resnet50/')
parser.add_argument('--result_path', type=str, default='./result.txt')
parser.add_argument('--log_step', type=int , default=10,
                    help='step size for prining log info')
parser.add_argument('--model_num', type=int , default=50,
                    help='step size for prining log info')

parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--maxlen', type=int, default=50)
parser.add_argument('--src_vocab_path', type=str, default='./models/src_vocab.pkl',
                    help='path for source vocabulary wrapper')
parser.add_argument('--tgt_vocab_path', type=str, default='./models/tgt_vocab.pkl',
                    help='path for target vocabulary wrapper')
parser.add_argument('--embed_size', type=int , default=620 ,
                    help='dimension of word embedding vectors')
parser.add_argument('--encoder_hidden_size', type=int , default=500 ,
                    help='dimension of encoder lstm hidden states')
parser.add_argument('--decoder_hidden_size', type=int, default=1000)
parser.add_argument('--num_layer', type=int, default=2)
parser.add_argument('--cuda_num', type=int, default=0)

args = parser.parse_args()

torch.cuda.set_device(args.cuda_num)
result_path = args.result_path
encoder_path = './models/encoder-' + args.attn + str(args.model_num) + '.pkl'
decoder_path = './models/decoder-' + args.attn + str(args.model_num) + '.pkl'

with open(args.src_vocab_path, 'rb') as f:
    src_vocab = pickle.load(f)
with open(args.tgt_vocab_path, 'rb') as f:
    tgt_vocab = pickle.load(f)

test_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                 batch_size=args.batch_size, type='test', shuffle=False)

encoder = Encoder(args.embed_size, args.encoder_hidden_size, src_vocab, num_layers=args.num_layers)
decoder = AttnDecoder_1(args.embed_size, args.decoder_hidden_size, tgt_vocab, args.image_feature_size)
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))


if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    print("Cuda is enabled...")

results = []

encoder.eval()
decoder.eval()
for bi, (sources, targets, src_lengths, tgt_lengths, image_features) in enumerate(test_data_loader):
    # sources: size = (1, max_len)
    sources = sources.to(device)
    targets = targets.to(device)
    image_features = image_features.to(device)
    encoder_output = encoder(sources)
    output = decoder.sample(encoder_output, image_features, args.maxlen)  # output有<start>和<end>，且是tensor的list
    ans = ''
    for i, id in enumerate(output[1:-1]):
        ans += tgt_vocab.idx2word[id.item()].decode() + ' '
    results.append(ans)

with open(args.result_path, 'w') as f:
    f.writelines(results)

