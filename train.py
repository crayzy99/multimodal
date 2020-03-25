# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import get_loader
from vocab import Vocabulary
from util import *
import os
import pickle
import random
import shutil


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(sources, targets, lengths, mask, encoder, decoder, encoder_optimizer,
          decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, teacher_forcing_ratio, image_features):
    '''
    :param sources: size = (max_src_len, batch), 注意与一般情形是相反的
    :param targets: size = (max_tgt_len, batch)
    :param lengths: 解码端句子的长度
    :param mask:
    :param encoder:
    :param decoder:
    :param embedding:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param src_vocab:
    :param tgt_vocab:
    :param args:
    :param max_target_len:
    :return:
    '''
    batch_size = args.batch_size
    clip = args.clip
    MAX_LEN = args.max_len

    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    sources = sources.to(device)
    targets = targets.to(device)
    lengths = lengths.to(device)
    image_features = image_features.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    # 前向传播

    encoder_outputs, encoder_hidden, image_features = encoder_forward(args.model_name, encoder,
                                                                      sources, lengths, image_features)
    decoder_input = torch.LongTensor([[tgt_vocab('<start>')] for _ in range(batch_size)])  ## size = (batch, 1)
    decoder_input = decoder_input.to(device)
    decoder_hidden = decoder.init_hidden(encoder_outputs,  encoder_hidden)
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Decoder逐步向前传播
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder_forward(args, decoder, decoder_input, decoder_hidden,
                                                             encoder_outputs, image_features)
            decoder_input = targets[t].view(-1, 1)  ## 对于teacher_forcing情形，下一个cell的输入从ground truth中获得
            masked_loss, nTotal = maskedNLLLoss(decoder_output, targets[:, t], mask[:, t])
            loss += masked_loss
            print_losses.append(masked_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder_forward(args, decoder, decoder_input, decoder_hidden,
                                                             encoder_outputs, image_features)

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskedNLLLoss(decoder_output, targets[:, t], mask[:, t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()
    encoder_total_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    decoder_total_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def validate(sources, targets, lengths, mask, encoder, decoder, encoder_optimizer,
          decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, image_features):
    '''
    :param sources: size = (max_src_len, batch), 注意与一般情形是相反的
    :param targets: size = (max_tgt_len, batch)
    :param lengths: 解码端句子的长度
    :param mask:
    :param encoder:
    :param decoder:
    :param embedding:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param src_vocab:
    :param tgt_vocab:
    :param args:
    :param max_target_len:
    :return:
    '''
    batch_size = 1

    sources = sources.to(device)
    targets = targets.to(device)
    lengths = lengths.to(device)
    image_features = image_features.to(device)
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    # 前向传播
    encoder_outputs, encoder_hidden, image_features = encoder_forward(args.model_name, encoder,
                                                                      sources, lengths, image_features)
    decoder_input = torch.LongTensor([[tgt_vocab('<start>')] for _ in range(batch_size)])  ## size = (batch, 1)
    decoder_input = decoder_input.to(device)
    decoder_hidden = decoder.init_hidden(encoder_outputs, encoder_hidden)
    batch_result = torch.zeros((batch_size, max_target_len))

    # Decoder逐步向前传播
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder_forward(args, decoder, decoder_input, decoder_hidden,
                                                         encoder_outputs, image_features)

        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0]] for i in range(batch_size)])  ## size = (batch,1)
        decoder_input = decoder_input.to(device)
        batch_result[t] = decoder_input
        # Calculate and accumulate loss
        mask_loss, nTotal = maskedNLLLoss(decoder_output, targets[:, t], mask[:, t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    return sum(print_losses) / n_totals, batch_result


def train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, src_embedding,tgt_embedding,
                src_vocab, tgt_vocab, args):
    batch_size  = args.batch_size
    print_every = args.print_every
    # load batches
    train_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                   batch_size=batch_size, type='train', shuffle=False,
                                   src_lg=args.src_language, tgt_lg=args.tgt_language)
    val_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                 batch_size=1, type='val', shuffle=False,
                                 src_lg=args.src_language, tgt_lg=args.tgt_language)

    # Initialization
    print('Initializing...')
    start_epoch = args.pretrained_epoch
    total_train_step = int(len(train_data_loader) * args.train_length)
    total_val_step = len(val_data_loader)
    print('Total train step:', total_train_step)
    print('Total val step:', total_val_step)
    min_avg_loss = float("inf")
    overfit_warn = 0
    print('Start training...')
    directory = os.path.join(args.model_path, args.model_name, '{}-{}'.format(args.src_language, args.tgt_language),
                              '{}layers_{}hidden'.format(args.num_layers, args.hidden_size))
    if os.path.exists(directory):
        shutil.rmtree(directory)
    for epoch in range(start_epoch, args.num_epochs+1):
        epoch_loss = 0
        print_loss = 0
        encoder.train()
        decoder.train()
        
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features, mask) in enumerate(train_data_loader):
            if bi > total_train_step:
                break
            max_target_len = torch.max(tgt_lengths)
            # TODO: 继续完成下面的训练步骤。思考问题：分布式训练；模型参数如何得到
            loss = train(sources, targets, src_lengths, mask, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, args.teacher_forcing_ratio,
                         image_features)
            print_loss += loss
            epoch_loss += loss

            if bi % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("[Epoch {}, train iteration {}] Average loss: {:.4f}".format(epoch, bi, print_loss_avg))
                print_loss = 0

        batch_loss_avg = epoch_loss / total_train_step
        print("[Epoch {}] Average train loss: {:.4f}".format(epoch, batch_loss_avg))

        # start evaluation step
        print("Evaluating for epoch {}...".format(epoch))
        eval_loss = 0
        print_loss = 0
        decoder_output = []
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features, mask) in enumerate(val_data_loader):
            encoder.eval()
            decoder.eval()
            max_target_len = torch.max(tgt_lengths)
            loss, batch_output = validate(sources, targets, src_lengths, mask, encoder, decoder, encoder_optimizer,  ## size = (batch_size, max_len)
                                            decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len,
                                          image_features)
            decoder_output.append(batch_output)
            print_loss += loss
            eval_loss += loss

            if bi % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("[Epoch {}, evaluation iteration {}] Average loss: {:.4f}".format(epoch, bi, print_loss_avg))
                print_loss = 0

        eval_avg_loss = eval_loss / total_val_step
        print("[Epoch {}] Average eval loss: {:.4f}".format(epoch, eval_avg_loss))
        #decoder_output = torch.cat(decoder_output, 0)  ## size = (val_size, max_len)
        #decoder_output = decoder_output.cpu().numpy()

        # generate answer for the first 10 sentences
        print("Decode sample:")
        for i in range(10):
            ans = ''
            tmp_output = decoder_output[i].cpu().numpy()[0]
            for i, idx in enumerate(tmp_output):
                if idx == tgt_vocab('<end>'):
                    break
                ans += tgt_vocab.idx2word[idx.item()] + ' '
            print(ans)

        # save model
        directory = os.path.join(args.model_path, args.model_name, '{}-{}'.format(args.src_language, args.tgt_language),
                                 '{}layers_{}hidden'.format(args.num_layers, args.hidden_size))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
            'epoch': epoch,
            'enc': encoder.state_dict(),
            'dec': decoder.state_dict(),
            'enc_opt': encoder_optimizer.state_dict(),
            'dec_opt': decoder_optimizer.state_dict(),
            'epoch_loss': batch_loss_avg,
            'source_embedding': src_embedding.state_dict(),
            'target_embedding': tgt_embedding.state_dict()
        }, os.path.join(directory, '{}_{}_{:.2f}.tar'.format(epoch, 'checkpoint', eval_avg_loss)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' % (epoch + 1)))
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' % (epoch + 1)))

        overfit_warn = overfit_warn + 1 if (min_avg_loss < eval_avg_loss) else 0
        min_avg_loss = min(min_avg_loss, eval_avg_loss)

        if overfit_warn >= 10:
            print("Model overfit!")
            break


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
        encoder_optimizer_sd = checkpoint['enc_opt']
        decoder_optimizer_sd = checkpoint['dec_opt']
        src_embedding_sd = checkpoint['source_embedding']
        tgt_embedding_sd = checkpoint['target_embedding']

    print('Building encoder and decoder...')
    src_embedding = nn.Embedding(len(src_vocab), args.embed_size)
    tgt_embedding = nn.Embedding(len(tgt_vocab), args.embed_size)
    if args.file_name:
        src_embedding.load_state_dict(src_embedding_sd)
        tgt_embedding.load_state_dict(tgt_embedding_sd)
    encoder, decoder = get_model(args, src_vocab, tgt_vocab, src_embedding, tgt_embedding)

#    ############## New code 3.14 ##############
#    if torch.cuda.device_count() > 1:
#        print("Let's use", torch.cuda.device_count(), "GPUs!")
#        encoder = nn.DataParallel(encoder)
#        decoder = nn.DataParallel(decoder)
#    ############## New code end  ##############

    if args.file_name:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('Model built and ready to go! The model has %d trainable parameters' % (count_parameters(encoder)+count_parameters(decoder)))

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)
    if args.file_name:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    print('Start training!')
    train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, src_embedding, tgt_embedding,
                src_vocab, tgt_vocab, args)


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
    args = parser.parse_args()
    print(args)
    main(args)
