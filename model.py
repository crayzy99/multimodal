# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchnlp.nn as nlpnn


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, embed_dropout_rate=0.4, output_dropout_rate=0.5,
                 num_layers=1):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size  # 400
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.GRU(embed_size, hidden_size,
                               batch_first=True, num_layers=num_layers, bidirectional=True)
        self.embedding_dropout = nn.Dropout(embed_dropout_rate)
        self.output_dropout = nn.Dropout(output_dropout_rate)

    def get_params(self):
        return list(self.encoder.parameters())

    def forward(self, src):
        '''
        :param src: size = (batch, src_max_len), dtype=long
        :param tgt: size = (batch, tgt_max_len), dtype=long
        :return: output: size = (batch, max_len, num_direction*hidden_size)
        '''
        embed = self.embedding(src)
        embed = self.embedding_dropout(embed)
        output, h_n = self.encoder(embed)  # (batch, seq_len, num_layers*hidden_size)
        output = self.output_dropout(output)

        return output


class AttnDecoder_1(nn.Module):
    '''
    第一种decoder方式，text和image互相做attention之后送入普通的decoder attention
    '''
    def __init__(self, embed_size, hidden_size, vocab, image_size):
        super(AttnDecoder_1, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size  # 不管了直接设置为input_size方便操作
        self.image_size = image_size
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.decoder_1 = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.decoder_2 = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.img2txt = nn.Linear(image_size, hidden_size)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)  # 这里也是，先不设置tying
        self.embedding.weight.requires_grad = False
        self.linear = nn.Linear(hidden_size, self.vocab_size)
        self.h1toh2 = nn.Linear(2*hidden_size, hidden_size)  # In the CONCAT connection method
        self.img_txt_attention = nlpnn.Attention(hidden_size)
        self.txt_img_attention = nlpnn.Attention(hidden_size)
        self.dec_img_attention = nlpnn.Attention(hidden_size)
        self.dec_txt_attention = nlpnn.Attention(hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.init_map = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(2)

    def get_params(self):
        params = list(self.decoder_1.parameters()) + list(self.decoder_2.parameters()) +\
                 list(self.img2txt.parameters()) + list(self.h1toh2.parameters()) +\
                 list(self.img_txt_attention.parameters()) + list(self.txt_img_attention.parameters()) +\
                 list(self.dec_img_attention.parameters()) + list(self.dec_txt_attention.parameters())
        return params

    def init_hidden(self, txt_features):
        tmp = torch.mean(txt_features, dim=1)
        tmp = self.tanh(self.init_map(tmp))
        return tmp.reshape(1, tmp.shape[0], self.hidden_size)

    def text_image_attention(self, text_features, image_features):
        '''
        :param text_features: shape = (batch, src_text_len, hidden_size)
        :param image_features: shape = (batch, image_len, hidden_size)
        :return: text_hat: shape = (batch, image_len, hidden_size)
                 image_hat: shape = (batch, src_text_len, hidden_size)
        这里的attention可以用concat和general两种方式，默认使用general
        '''
        image_features = image_features.contiguous()
        text_features = text_features.contiguous()
        text_hat, _ = self.img_txt_attention(image_features, text_features)
        image_hat, _ = self.txt_img_attention(text_features, image_features)

        return text_hat, image_hat

    def forward(self, text_input, image_input, tgt):
        '''
        :param text_input: shape = (batch, src_text_len, hidden_size)
        :param image_input: shape = (batch, image_len, image_size)
        :param tgt: shape = (batch, tgt_text_len,)
        :return:
        '''
        batch_size = text_input.shape[0]
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        embed = self.embedding(tgt_input)
        h_0 = self.init_hidden(text_input)
        decoder1_output, h_n = self.decoder_1(embed, h_0)  # shape = (batch, tgt_txt_len, hidden_size)
        image_input = self.img2txt(image_input)  # shape = (batch, image_len, hidden_size)
        text_hat, image_hat = self.text_image_attention(text_input, image_input)
        decoder1_output = decoder1_output.contiguous()
        text_hat = text_hat.contiguous()
        image_hat = image_hat.contiguous()
        text_bar, _ = self.dec_txt_attention(decoder1_output, text_hat)
        image_bar, _ = self.dec_img_attention(decoder1_output, image_hat)
        input_2 = self.h1toh2(torch.cat([text_bar, image_bar], dim=2))  # shape = (batch, tgt_txt_len, hidden_size)
        h_0 = decoder1_output[:, 1]
        h_0 = h_0.reshape(1, batch_size, self.hidden_size).contiguous()
        decoder2_output, h_n = self.decoder_2(input_2, h_0)
        output = self.linear(decoder2_output)  # shape = (batch, tgt_txt_len, vocab_size)

        return output, tgt_output

    def sample(self, text_input, image_input, maxlen, ):
        '''
        :param text_input: shape = (batch, src_max_len, hidden_size)
        :param image_input: size = (batch, image_len, image_size)
        :return: output: size =
        论文中说用h1(1)来初始化h0(2)，但是sample的时候h1(1)还没出来就要用h0(2)了？先用相同的初始化方式了。
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vocab = self.vocab
        idx = torch.LongTensor([vocab(b'<start>')]).to(device)
        input = self.embedding(idx).view(1,1,-1)
        h_1 = self.init_hidden(text_input)
        h_2 = self.init_hidden(text_input)
        batch_size = text_input.shape[0]
        image_input = self.img2txt(image_input)
        text_hat, image_hat = self.text_image_attention(text_input, image_input)
        text_hat = text_hat.contiguous()
        image_hat = image_hat.contiguous()

        ans = []
        while True:
            decoder1_output, h_1 = self.decoder_1(input, h_1)  # (1, 1, hidden_size)
            text_bar, _ = self.dec_txt_attention(decoder1_output, text_hat)  # shape = (1, 1, hidden_size)
            image_bar, _ = self.dec_img_attention(decoder1_output, image_hat)
            input_2 = self.h1toh2(torch.cat([text_bar, image_bar], dim=2))
            decoder2_output, h_2 = self.decoder_2(input_2, h_2)
            decoder2_output = decoder1_output.squeeze().squeeze()
            max, id = torch.max(decoder2_output, 0)
            ans.append(id)
            if vocab.idx2word[id.item()] == b'<end>':
                ans = ans[:-1]
                break
            if len(ans) >= maxlen:
                break
            input = self.embedding(id).view(1,1,-1)

        ans = torch.stack(ans, 0)
        return ans
