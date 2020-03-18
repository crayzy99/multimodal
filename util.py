import torch
import torch.optim as optim
import torch.nn as nn
from model import EncoderGRU, LuongAttention, TextAttnDecoderGRU, DIRECTDecoder, DIRECTEncoder

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_optimizer(optimizer, lr, params, weight_decay):
    if optimizer == 'sgd':
        return optim.SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer == 'adam':
        return optim.Adam(params, lr, weight_decay=weight_decay)


def get_activation(activation):
    if not activation:
        return None
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'prelu':
        return nn.PReLU()
    elif activation == 'lrelu':
        return nn.LeakyReLU()
    elif activation == 'tanh':
        return nn.Tanh()


class RawDataset:
    def __init__(self, data_path, types, src_language, tgt_language):
        ## 此为构造词表所需，为方便起见把训练集也加入词表构建了
        self.src = []
        self.tgt = []
        for t in types:
            if t == 'test':
                src_file = data_path + t + '_2016_flickr.lc.norm.tok.' + src_language
                tgt_file = data_path + t + '_2016_flickr.lc.norm.tok.' + tgt_language
            else:
                src_file = data_path + t + '.lc.norm.tok.' + src_language
                tgt_file = data_path + t + '.lc.norm.tok.' + tgt_language
            with open(src_file, 'r', encoding='utf-8') as f:
                self.src += f.readlines()
            with open(tgt_file, 'r', encoding='utf-8') as f:
                self.tgt += f.readlines()

        for i, line in enumerate(self.tgt):
            if line[-1] == '\n':
                self.tgt[i] = line[:-1]

        for i, line in enumerate(self.src):
            if line[-1] == '\n':
                self.src[i] = line[:-1]

def maskedNLLLoss(out, target, mask):
    '''
    :param out: size = (batch, output_size), prob_like
    :param target: size = (batch), token index
    :param mask: size = (batch), 0 or 1
    :return: masked loss: size = (batch)
    '''
    nTotal = mask.sum()
    crossEntropyLoss = -torch.log(torch.gather(out, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropyLoss.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def binaryMatrix(l, value=0):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == value:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def encoder_forward(model_name, encoder, sources, lengths, image_features):
    if model_name == 'seq2seq-text':
        encoder_outputs, encoder_hidden, image_features = encoder(sources, lengths)
    elif model_name == 'DIRECT':
        encoder_outputs, encoder_hidden, image_features = encoder(sources, lengths, image_features)

    return encoder_outputs, encoder_hidden, image_features


def decoder_forward(decoder_input, decoder_hidden,
                    encoder_outputs, image_features)
    if args.model_name == 'TEXT':
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
    elif args.model_name == 'DIRECT':
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs, image_features
        )

    return decoder_output, decoder_hidden

def get_model(args):
    if args.model_name == 'seq2seq2-text':
        encoder = EncoderGRU(args.embed_size, args.hidden_size, src_vocab, src_embedding, args.embedding_dropout_rate,
                         args.output_dropout_rate, args.num_layers)
        decoder = TextAttnDecoderGRU(args.attn_model, tgt_embedding, args.hidden_size, len(tgt_vocab), args.num_layers,
                                     args.output_dropout_rate)
    elif args.model_name == 'DIRECT':
        encoder = DIRECTEncoder(args.embed_size, args.hidden_size, src_vocab, src_embedding, args.embedding_dropout_rate,
                         args.output_dropout_rate, args.num_layers)
        decoder = DIRECTDecoder(args.attn_model, tgt_embedding, args.hidden_size, len(tgt_vocab), args.embed_size)

    return encoder, decoder

