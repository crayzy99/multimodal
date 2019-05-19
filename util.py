import torch
import torch.optim as optim
import torch.nn as nn

def get_optimizer(optimizer, lr, params):
    if optimizer == 'sgd':
        return optim.SGD(params, lr, momentum=0.9)
    elif optimizer == 'adam':
        return optim.Adam(params, lr)

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

class En_De:
    def __init__(self, data_path, types, image_feature_dir):
        self.src = []
        self.tgt = []
        for t in types:
            if t == 'test':
                src_file = data_path + t + '_2016_flickr.lc.norm.tok.en'
                tgt_file = data_path + t + '_2016_flickr.lc.norm.tok.de'
            else:
                src_file = data_path + t + '.lc.norm.tok.en'
                tgt_file = data_path + t + '.lc.norm.tok.de'
            with open(src_file, 'rb') as f:
                self.src += f.readlines()
            with open(tgt_file, 'rb') as f:
                self.tgt += f.readlines()

        for i, line in enumerate(self.tgt):
            if line[-1] == '\n':
                self.tgt[i] = line[:-1]

        for i, line in enumerate(self.src):
            if line[-1] == '\n':
                self.src[i] = line[:-1]
