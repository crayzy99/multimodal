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
    def __init__(self, image_dir, sis_path, vocab, type):
        self.image_dir = image_dir
        self.vocab = vocab
        self.image_features = pickle.load(open(image_dir, 'rb'))

    def __getitem__(self, index):
        vist = self.vist
        vocab = self.vocab
        #story_id = self.ids[index]

        targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[index]
        image_ids = vist.images[index]
        for id in image_ids:
            images.append(torch.Tensor(self.image_features[id]))
            photo_sequence.append(id)
        for annotation in story:
            tokens = annotation.split(' ')
            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        # story = vist.stories[story_id]
        # image_formats = ['.jpg', '.gif', '.png', '.bmp']
        # for annotation in story:
        #     storylet_id = annotation["storylet_id"]
        #     image = Image.new('RGB', (256, 256))
        #     image_id = annotation["photo_flickr_id"]
        #     photo_sequence.append(image_id)
        #     album_ids.append(annotation["album_id"])
        #     for image_format in image_formats:
        #         try:
        #             image = Image.open(os.path.join(self.image_dir, str(image_id) + image_format)).convert('RGB')
        #         except Exception:
        #             continue
        #
        #     if self.transform is not None:
        #         image = self.transform(image)
        #
        #     images.append(image)
        #
        #     text = annotation["text"]
        #     tokens = []
        #     try:
        #         tokens = nltk.tokenize.word_tokenize(text.lower())
        #     except Exception:
        #         pass
        #
        #     caption = []
        #     caption.append(vocab('<start>'))
        #     caption.extend([vocab(token) for token in tokens])
        #     caption.append(vocab('<end>'))
        #     target = torch.Tensor(caption)
        #     targets.append(target)

        return torch.stack(images), targets, photo_sequence, album_ids


    def __len__(self):
        return len(self.vist.stories)

    def GetItem(self, index):
        vist = self.vist
        vocab = self.vocab
        story_id = self.ids[index]

        targets = []
        images = []
        photo_sequence = []
        album_ids = []

        story = vist.stories[story_id]
        image_formats = ['.jpg', '.gif', '.png', '.bmp']
        for annotation in story:
            storylet_id = annotation["storylet_id"]
            image = Image.new('RGB', (256, 256))
            image_id = annotation["photo_flickr_id"]
            photo_sequence.append(image_id)
            album_ids.append(annotation["album_id"])
            for image_format in image_formats:
                try:
                    image = Image.open(os.path.join(self.image_dir, image_id + image_format)).convert('RGB')
                    break
                except Exception:
                    continue

            if self.transform is not None:
                image = self.transform(image)

            images.append(image)

            text = annotation["text"]
            tokens = []
            try:
                tokens = nltk.tokenize.word_tokenize(text.lower())
            except Exception:
                pass

            caption = []
            caption.append(vocab('<start>'))
            caption.extend([vocab(token) for token in tokens])
            caption.append(vocab('<end>'))
            target = torch.Tensor(caption)
            targets.append(target)

        return images, targets, photo_sequence, album_ids

    def GetLength(self):
        return len(self.ids)


def collate_fn(data):

    image_stories, caption_stories, photo_sequence_set, album_ids_set = zip(*data)

    targets_set = []
    lengths_set = []

    for captions in caption_stories:
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        targets_set.append(targets)
        lengths_set.append(lengths)

    return image_stories, targets_set, lengths_set, photo_sequence_set, album_ids_set


def get_loader(root, sis_path, vocab, batch_size, shuffle, num_workers, type):
    vist = VistDataset(image_dir=root, sis_path=sis_path, vocab=vocab, type=type)

    data_loader = torch.utils.data.DataLoader(dataset=vist, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    return data_loader