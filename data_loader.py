import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from vist import VIST


class Flickr30k(data.Dataset):
    def __init__(self, image_feature_dir, data_path, type, src_vocab, tgt_vocab):
        self.image_feature_dir = ''
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src = []
        self.tgt = []
        if type == 'test':
            src_file = data_path + type + '_2016_flickr.lc.norm.tok.en'
            tgt_file = data_path + type + '_2016_flickr.lc.norm.tok.de'
            self.image_feature_dir = image_feature_dir + t + '_2016_flickr-resnet50-avgpool.npy'
        else:
            src_file = data_path + type + '.lc.norm.tok.en'
            tgt_file = data_path + type + '.lc.norm.tok.de'
            self.image_feature_dir = image_feature_dir + t + '-resnet50-avgpool.npy'
        with open(src_file, 'rb') as f:
            self.src += f.readlines()
        with open(tgt_file, 'rb') as f:
            self.tgt += f.readlines()

        self.image_features = np.load(self.image_feature_dir)

        print('First line in src_file:', self.src[0])
        print('First line in tgt_file:', self.tgt[0])
        print('Image feature size:', self.image_features.shape)

    def __getitem__(self, index):
        src_vocab = self.src_vocab
        tgt_vocab = self.tgt_vocab
        src = self.src
        tgt = self.tgt
        image_features = self.image_features

        src_sent = src[index]
        tgt_sent = tgt[index]
        source = []
        target = []
        image_feature = image_features[index]

        src_tokens = src_sent.split()
        source.append(src_vocab(b'<start>'))
        source.extend([src_vocab(token) for token in src_tokens])
        source.append(src_vocab(b'<end>'))
        source = torch.Tensor(source)

        tgt_tokens = tgt_sent.split()
        target.append(tgt_vocab(b'<start>'))
        target.extend([tgt_vocab(token) for token in tgt_tokens])
        target.append(tgt_vocab(b'<end>'))
        target = torch.Tensor(target)

        return source, target, image_feature

    def __len__(self):
        return len(self.vist.stories)


def collate_fn(data):
    sources, targets, image_features = zip(*data)
    src_lengths = [len(source) for source in sources]
    sources_tensor = torch.zeros(len(sources), max(src_lengths)).long()
    for i, source in enumerate(sources):
        end = src_lengths[i]
        sources_tensor[i, :end] = source

    tgt_lengths = [len(target) for target in targets]
    targets_tensor = torch.zeros(len(targets), max(tgt_lengths)).long()
    for i, target in enumerate(targets):
        end = tgt_lengths[i]
        targets_tensor[i, :end] = target

    image_features = torch.stack(image_features, dim=0)

    return sources_tensor, targets_tensor, src_lengths, tgt_lengths, image_features


def get_loader(image_feature_dir, data_path, src_vocab, tgt_vocab, batch_size, type, shuffle):
    dataset = Flickr30k(image_feature_dir, data_path, type, src_vocab, tgt_vocab)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader
