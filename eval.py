import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import get_loader
import torch.nn.functional as F
from vocab import Vocabulary
from util import *
from model import EncoderGRU, LuongAttention, TextAttnDecoderGRU
import os
import pickle
import random
import operator

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def path2sentence(path, vocab):
    ans_str = ''
    for i, idx in enumerate(path):
        if idx == vocab('<end>'):
            break
        ans_str += vocab.idx2word[idx.item()] + ' '
    return ans_str

def evaluate(encoder, decoder, src_embedding, tgt_embedding, src_vocab, tgt_vocab, args):
    test_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                   batch_size=1, type='val', shuffle=False,
                                   src_lg=args.src_language,
                                   tgt_lg=args.tgt_language)

    print("Total test examples:", len(test_data_loader))
    print("Start testing!")
    encoder.eval()
    decoder.eval()
    test_loss = 0.0
    results = []
    output = []
    for bi, (sources, targets, src_lengths, tgt_lengths, image_features, mask) in enumerate(test_data_loader):
        if bi >= 10:
            break
        max_target_len = torch.max(tgt_lengths)
        batch_size = 1

        sources = sources.to(device)
        lengths = src_lengths.to(device)

        # 前向传播
        encoder_outputs, encoder_hidden, image_features = encoder_forward(args.model_name, encoder,
                                                                          sources, lengths, image_features)
        decoder_hidden = decoder.init_hidden(encoder_outputs, encoder_hidden)

        # prev_paths保存目前概率最高的k个path（初始只有一个）. 列表元素有两个，第一个是一个tuple保存路径（也是一个列表）以及
        # 生成下一个token所需的hidden_state
        vocab = tgt_vocab
        idx = torch.LongTensor([vocab('<start>')]).to(device)
        prev_paths = [[([idx], decoder_hidden), 1.0]]
        # new_paths保存下一个token可能的情况，计算完成后将会有k**2个条目，
        new_paths = []
        beam_size = args.beam_size
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]
        
        # Decoder逐步向前传播
        for t in range(max_target_len):
            # beam search
            for i, prev_ in enumerate(prev_paths):
                prev_condition = prev_[0]  ## 也就是之前的path和当前hidden state
                prob = prev_[1]  # 当前path概率
                prev_path, prev_hidden = prev_condition
                last_word_idx = prev_path[-1]
                # 如果path中的上一个token是eof，直接break
                if last_word_idx == end_vocab:
                    new_paths.append(prev_paths[i])
                    break
                # 过decoder
                decoder_input = torch.LongTensor([[last_word_idx]])
                decoder_input = decoder_input.to(device)
                decoder_output, decoder_hidden = decoder_forward(decoder_input, decoder_hidden,
                                                                 encoder_outputs, image_features)
                decoder_output = decoder_output.squeeze().squeeze()  # shape = (hidden_size,) 未经softmax
                # 如果上一个词不是终止token（标点符号），则禁止输出eof
                if last_word_idx not in termination_list:
                    decoder_output[end_vocab] = -1000.0
                # 禁止输出fobidden tokens
                for forbidden in forbidden_list:
                    decoder_output[forbidden] = -1000.0
                if last_word_idx in function_list:
                    for word in function_list:
                        decoder_output[word] = -1000.0

                output_prob = F.softmax(decoder_output, dim=0)  # shape = (hidden_size,) prob_like
                values, indices = torch.topk(output_prob, beam_size)
                for ix in range(beam_size):
                    new_paths.append([(prev_path + [indices[ix]], decoder_hidden), prob + torch.log(values[ix] + 1e-6)])
        
            # 现在new_paths里应该有beam_size**2个条目
            sorted_paths = sorted(new_paths, key=operator.itemgetter(1),
                    reverse=True)
            prev_paths = sorted_paths[:beam_size]
#            print("The picked paths of step {}:".format(t))
#            for i in range(len(sorted_paths)):
#                print(path2sentence(prev_paths[i][0][0], tgt_vocab))
            new_paths = []
        
        assert len(prev_paths) > 0
        ans = prev_paths[0][0][0]

        print("Decoding sample {}/{}".format(bi, len(test_data_loader)))
        ans_str = path2sentence(ans, tgt_vocab)
        results.append(ans_str)
        
    with open("results.txt", "w") as f:
         f.writelines([line+"\n" for line in results])

#        batch_output = torch.zeros(max_target_len)
#        for t in range(max_target_len):
#            decoder_output, decoder_hidden = decoder(
#                decoder_input, decoder_hidden, encoder_outputs
#            )
#            # No teacher forcing: next input is decoder's own current output
#            _, topi = decoder_output.topk(1)
#            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])  ## size = (1,batch)
#            decoder_input = decoder_input.to(device)
#            batch_output[t] = decoder_input[0]
#        output.append(batch_output)
#
#    for i in range(10):
#        ans = ''
#        tmp_output = output[i].cpu().numpy()
#        for i, idx in enumerate(tmp_output):
#            if idx == tgt_vocab('<end>'):
#                break
#            ans += tgt_vocab.idx2word[idx.item()] + ' '
#        print(ans)


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    vocab_path = args.model_path + args.src_language + '-' + args.tgt_language + '/'
    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)
    src_vocab_path = vocab_path + 'src_vocab.pkl'
    tgt_vocab_path = vocab_path + 'tgt_vocab.pkl'

    with open(src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(tgt_vocab_path, 'rb') as f:
        tgt_vocab = pickle.load(f)
    torch.cuda.set_device(args.cuda_num)

    if args.file_name:
        checkpoint = torch.load(args.file_name)
        encoder_sd = checkpoint['enc']
        decoder_sd = checkpoint['dec']
        src_embedding_sd = checkpoint['source_embedding']
        tgt_embedding_sd = checkpoint['target_embedding']
    print('Building encoder and decoder...')
    src_embedding = nn.Embedding(len(src_vocab), args.embed_size)
    tgt_embedding = nn.Embedding(len(tgt_vocab), args.embed_size)
    print(len(src_vocab))
    if args.file_name:
        src_embedding.load_state_dict(src_embedding_sd)
        tgt_embedding.load_state_dict(tgt_embedding_sd)
    encoder, decoder = get_model(args, src_vocab, tgt_vocab, src_embedding, tgt_embedding)
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
    parser.add_argument('--model_name', type=str, default='TEXT')
    parser.add_argument('--attn_model', type=str, default='dot')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=1)
    parser.add_argument('--train_length', type=float, default=1)
    parser.add_argument('--beam_size', type=int, default=12)

    args = parser.parse_args()
    main(args)
