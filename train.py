# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import get_loader
from vocab import Vocabulary
from util import *
from model import Encoder, AttnDecoder_1
import os
import pickle


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(args.src_vocab_path, 'rb') as f:
        src_vocab = pickle.load(f)
    with open(args.tgt_vocab_path, 'rb') as f:
        tgt_vocab = pickle.load(f)

    torch.cuda.set_device(args.cuda_num)

    train_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                   batch_size=args.batch_size, type='train', shuffle=True)
    val_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                 batch_size=args.batch_size, type='val', shuffle=True)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('The model has %d trainable parameters' % (count_parameters(encoder)+count_parameters(decoder)))

    pretrained_epoch = 0
    if args.pretrained_epoch > 0:
        pretrained_epoch = args.pretrained_epoch
        encoder.load_state_dict(torch.load('./models/encoder-' + str(pretrained_epoch) + '.pkl'))
        decoder.load_state_dict(torch.load('./models/decoder-' + str(pretrained_epoch) + '.pkl'))  # 先不用多种attention方法了

    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    criterion = nn.CrossEntropyLoss()
    params = encoder.get_params() + decoder.get_params()
    optimizer = get_optimizer(args.optimizer, args.learning_rate, params, weight_decay=args.L2_lambda)

    total_train_step = len(train_data_loader)
    total_val_step = len(val_data_loader)

    min_avg_loss = float("inf")
    overfit_warn = 0

    for epoch in range(args.num_epochs):

        if epoch < pretrained_epoch:
            continue

        encoder.train()
        decoder.train()
        avg_loss = 0.0
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features) in enumerate(train_data_loader):
            encoder.zero_grad()
            decoder.zero_grad()
            loss = 0.0
            try:
                sources = sources.to(device)
                targets = targets.to(device)
                image_features = image_features.to(device)
                encoder_output = encoder(sources)
                output, tgt_output = decoder(encoder_output, image_features, targets)
                output = output.contiguous().view(-1, output.shape[-1])
                tgt_output = tgt_output.contiguous().view(-1)
                loss += criterion(output, tgt_output)

                avg_loss += loss.item()
                #loss /= args.batch_size
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.clip)
                optimizer.step()
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Train Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch + 1, args.num_epochs, bi, total_train_step,
                        loss.item(), np.exp(loss.item())))

            # if bi >= 20:
            #     break

        avg_loss /= total_train_step
        print('Epoch [%d/%d], Average Train Loss: %.4f' %
              (epoch + 1, args.num_epochs, avg_loss))

        if epoch % args.save_step == 0:
            torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' % (epoch+1)))
            torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' % (epoch + 1)))

        encoder.eval()
        decoder.eval()
        avg_loss = 0.0
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features) in enumerate(val_data_loader):
            sources = sources.to(device)
            targets = targets.to(device)
            image_features = image_features.to(device)
            loss = 0.0
            try:
                encoder_output = encoder(sources)
                output, tgt_output = decoder(encoder_output, image_features, targets)
                output = output.contiguous().view(-1, output.shape[-1])
                tgt_output = tgt_output.contiguous().view(-1)
                loss += criterion(output, tgt_output)

                avg_loss += loss.item()
                #loss /= args.batch_size
            except RuntimeError as exception:
                if "out of memory" in str(exception):
                    print("WARNING: out of memory")
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                else:
                    raise exception

            if bi % args.log_step == 0:
                print('Epoch [%d/%d], Val Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch + 1, args.num_epochs, bi, total_val_step,
                        loss.item(), np.exp(loss.item())))

            # if bi >= total_val_step:
            #     break

        avg_loss /= total_val_step
        print('Epoch [%d/%d], Average Val Loss: %.4f' %
              (epoch + 1, args.num_epochs, avg_loss))

        overfit_warn = overfit_warn + 1 if (min_avg_loss < avg_loss) else 0
        min_avg_loss = min(min_avg_loss, avg_loss)

        if overfit_warn >= 10:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--src_vocab_path', type=str, default='./models/src_vocab.pkl')
    parser.add_argument('--tgt_vocab_path', type=str, default='./models/tgt_vocab.pkl')
    parser.add_argument('--data_path', type=str, default='../task1/tok/')
    parser.add_argument('--image_feature_dir', type=str, default='../features_resnet50/')
    parser.add_argument('--log_step', type=str, default=20)
    parser.add_argument('--save_step', type=str, default=1)
    parser.add_argument('--embed_size', type=int, default=620)
    parser.add_argument('--encoder_hidden_size', type=int, default=500)
    parser.add_argument('--decoder_hidden_size', type=int, default=1000)
    parser.add_argument('--image_feature_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--embedding_dropout', type=float, default=0.4)
    parser.add_argument('--output_dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--L2_lambda', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=1)

    parser.add_argument('--pretrained_epoch', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--cuda_num', type=int, default=0)
    args = parser.parse_args()
    print(args)
    main(args)
