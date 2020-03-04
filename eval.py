import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import get_loader
from vocab import Vocabulary
from util import *
from model import EncoderGRU, LuongAttention, TextAttnDecoderGRU
import os
import pickle
import random

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(encoder, decoder, src_embedding, tgt_embedding, src_vocab, tgt_vocab, data_loader, args):
    

def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(args.src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.tgt_vocab_path, 'rb') as f:
        tgt_vocab = pickle.load(f)
    torch.cuda.set_device(args.cuda_num)

    if args.file_name:
        checkpoint = torch.load(args.file_name)
        encoder_sd = checkpoint['enc']
        decoder_sd = checkpoint['dec']
        src_embedding_sd = checkpoint['source_embedding']
        tgt_embedding_sd = checkpoint['target_embedding']
    print('Building encoder and decoder...')
    src_embedding = nn.Embedding(en(src_vocab), args.embed_size)
    tgt_embedding = nn.Embedding(len(tgt_vocab), args.embed_size)
    if args.file_name:
        src_embedding.load_state_dict(src_embedding_sd)
        tgt_embedding.load_state_dict(tgt_embedding_sd)
    encoder = EncoderGRU(args.embed_size, args.hidden_size, src_vocab, src_embedding, args.embedding_dropout_rate,
                         args.output_dropout_rate, args.num_layers)
    decoder = TextAttnDecoderGRU(args.attn_model, tgt_embedding, args.hidden_size, len(tgt_vocab), args.num_layers,
                                 args.output_dropout_rate)
    if args.file_name:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    print('Start evaluating!')
    evaluate(encoder, decoder, src_embedding, tgt_embedding, src_vocab, tgt_vocab, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--src_vocab_path', type=str, default='./models/src_vocab.pkl')
    parser.add_argument('--tgt_vocab_path', type=str, default='./models/tgt_vocab.pkl')
    parser.add_argument('--data_path', type=str, default='dataset/data/task1/tok/')
    parser.add_argument('--image_feature_dir', type=str, default='dataset/data/features_resnet50/')
    parser.add_argument('--log_step', type=str, default=20)
    parser.add_argument('--save_step', type=str, default=1)
    parser.add_argument('--embed_size', type=int, default=620)
    parser.add_argument('--hidden_size', type=int, default=500)
    #parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--image_feature_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--embedding_dropout_rate', type=float, default=0.4)
    parser.add_argument('--output_dropout_rate', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--L2_lambda', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--max_len', type=int, default=30)
    
    parser.add_argument('--model_name', type=str, default='seq2seq-text')
    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda_num', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--src_language', type=str, default='en')
    parser.add_argument('--tgt_language', type=str, default='de')
    parser.add_argument('--file_name', type=str, default=None)
    parser.add_argument('--attn_model', type=str, default='dot')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1)
    parser.add_argument('--train_length', type=float, default=1)
    parser.add_argument('--beam_size', type=int, default=12)
    
    args = parser.parse_args()
    main(args)

torch.cuda.set_device(args.cuda_num)
result_path = args.result_path
encoder_path = './models/encoder-' + str(args.model_num) + '.pkl'
decoder_path = './models/decoder-' + str(args.model_num) + '.pkl'

with open(args.src_vocab_path, 'rb') as f:
    src_vocab = pickle.load(f)
with open(args.tgt_vocab_path, 'rb') as f:
    tgt_vocab = pickle.load(f)

test_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                 batch_size=args.batch_size, type='test', shuffle=False)

encoder = Encoder(args.embed_size, args.encoder_hidden_size, src_vocab)
decoder = AttnDecoder_1(args.embed_size, args.encoder_hidden_size, tgt_vocab, args.image_size)
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
    output = decoder.sample(encoder_output, image_features, args.maxlen, args.beam_size)  # output有<start>和<end>，且是tensor的list
    ans = ''
    for i, id in enumerate(output[1:-1]):
        ans += tgt_vocab.idx2word[id.item()].decode() + ' '
    results.append(ans)

with open(args.result_path, 'w') as f:
    f.writelines(results)

