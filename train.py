# -*- coding:utf-8 -*-

import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from data_loader import get_loader
from vocab import Vocabulary
from util import *
from model import EncoderGRU, LuongAttention, TextAttnDecoderGRU
from eval import evaluate
import os
import pickle
import random


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(sources, targets, lengths, mask, encoder, decoder, encoder_optimizer,
          decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, teacher_forcing_ratio):
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
    mask = mask.to(device)

    loss = 0
    print_losses = []
    n_totals = 0

    # 前向传播
    encoder_outputs, encoder_hidden = encoder(sources, lengths)
    decoder_input = torch.LongTensor([[tgt_vocab('b<start>') for _ in range(batch_size)]])  ## size = (1, batch)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Decoder逐步向前传播
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_input = targets[t].view(1, -1)  ## 对于teacher_forcing情形，下一个cell的输入从ground truth中获得
            masked_loss, nTotal = maskedNLLLoss(decoder_output, targets[t], mask[t])
            loss += masked_loss
            print_losses.append(masked_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    loss.backward()
    encoder_total_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    decoder_total_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    return sum(print_losses) / n_totals


def train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, src_embedding,tgt_embedding,
                src_vocab, tgt_vocab, args):
    batch_size  = args.batch_size
    print_every = args.print_every
    # load batches
    train_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                   batch_size=args.batch_size, type='train', shuffle=True)
    val_data_loader = get_loader(args.image_feature_dir, args.data_path, src_vocab, tgt_vocab,
                                 batch_size=args.batch_size, type='val', shuffle=True)

    # Initialization
    print('Initializing...')
    start_epoch = args.pretrained_epoch
    total_train_step = len(train_data_loader)
    total_val_step = len(val_data_loader)
    print('Start training...')
    for epoch in range(start_epoch, args.num_epochs+1):
        epoch_loss = 0
        print_loss = 0
        encoder.train()
        decoder.train()
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features, mask) in enumerate(train_data_loader):
            max_target_len = torch.max(tgt_lengths)
            # TODO: 继续完成下面的训练步骤。思考问题：分布式训练；模型参数如何得到
            loss = train(sources, targets, tgt_lengths, mask, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, args.teacher_forcing_ratio)
            print_loss += loss
            epoch_loss += loss

            if bi % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("[Epoch {}, train iteration {}] Average loss: {:.4f}".format(epoch, bi, print_loss_avg))
                print_loss = 0

        batch_loss_avg = batch_loss / total_train_step
        print("[Epoch {}] Average train loss: {:.4f}".format(epoch, batch_loss_avg))

        # start evaluation step
        print("Evaluating for epoch {}...".format(epoch))
        eval_loss = 0
        print_loss = 0
        for bi, (sources, targets, src_lengths, tgt_lengths, image_features, mask) in enumerate(val_data_loader):
            encoder.eval()
            decoder.eval()
            max_target_len = torch.max(tgt_lengths)
            loss = train(sources, targets, tgt_lengths, mask, encoder, decoder, encoder_optimizer,
                         decoder_optimizer, src_vocab, tgt_vocab, args, max_target_len, 0)
            print_loss += loss
            eval_loss += loss

            if bi % print_every == 0:
                print_loss_avg = print_loss / print_every
                print("[Epoch {}, evaluation iteration {}] Average loss: {:.4f}".format(epoch, bi, print_loss_avg))
                print_loss = 0

        eval_loss_avg = eval_loss / total_val_step
        print("[Epoch {}] Average eval loss: {:.4f}".format(epoch, eval_loss_avg))

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
            'source_embedding': src_embedding,
            'target_embedding': tgt_embedding
        }, os.path.join(directory, '{}_{}_{:.2f}.tar'.format(epoch, 'checkpoint', eval_loss_avg)))
        torch.save(encoder.state_dict(), os.path.join(args.model_path, 'encoder-%d.pkl' % (epoch + 1)))
        torch.save(decoder.state_dict(), os.path.join(args.model_path, 'decoder-%d.pkl' % (epoch + 1)))


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
        encoder_optimizer_sd = checkpoint['enc_opt']
        decoder_optimizer_sd = checkpoint['dec_opt']
        src_embedding_sd = checkpoint['source_embedding']
        tgt_embedding_sd = checkpoint['target_embedding']

    src_embedding = nn.Embedding(len(src_vocab), args.embed_size)
    tgt_embedding = nn.Embedding(len(tgt_vocab), args.embed_size)
    if args.file_name:
        src_embedding.load_state_dict(src_embedding_sd)
        tgt_embedding.load_state_dict(tgt_embedding_sd)
    encoder = EncoderGRU(args.embed_size, args.hidden_size, src_vocab, src_embedding, args.embedding_dropout_rate,
                         args.output_dropout_rate, args.num_layers)
    decoder = TextAttnDecoderGRU(args.attn_model, tgt_embedding, args.hidden_size, len(tgt_vocab), args.num_layers,
                                 args.output_dropout_rate)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('The model has %d trainable parameters' % (count_parameters(encoder)+count_parameters(decoder)))

    print('Building encoder and decoder...')
    embedding = nn.Embedding()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if args.file_name:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    print('Start training!')
    train_iters(encoder, decoder, encoder_optimizer, decoder_optimizer, src_embedding, tgt_embedding,
                src_vocab, tgt_vocab, args)

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
    parser.add_argument('--hidden_size', type=int, default=500)
    #parser.add_argument('--hidden_size', type=int, default=1000)
    parser.add_argument('--image_feature_size', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--embedding_dropout', type=float, default=0.4)
    parser.add_argument('--output_dropout', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--L2_lambda', type=float, default=1e-5)
    parser.add_argument('--clip', type=float, default=1)
    parser.add_argument('--max_len', type=int, default=30)

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
    args = parser.parse_args()
    print(args)
    main(args)
