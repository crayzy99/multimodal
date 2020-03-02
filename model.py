# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torchnlp.nn as nlpnn
from collections import OrderedDict
from operator import itemgetter
import torch.nn.functional as F


class EncoderGRU(nn.Module):
    ## 一个简单的LSTM模型
    ## 可能的改进方向：加入BPE或者german compound word splitting.
    def __init__(self, embed_size, hidden_size, vocab, embedding, embed_dropout_rate=0.4, output_dropout_rate=0.5,
                 num_layers=1):
        super(EncoderGRU, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embedding.weight.requires_grad = False
        self.gru = nn.GRU(embed_size, hidden_size,
                              num_layers=num_layers, bidirectional=True)
        self.embedding_dropout = nn.Dropout(embed_dropout_rate)
        self.output_dropout = nn.Dropout(output_dropout_rate)

    def get_params(self):
        return list(self.gru.parameters())

    def forward(self, src, src_lengths, hidden=None):
        '''
        :param src: size = (src_max_len, batch), dtype=long
        :param src_lengths: size = (batch), batch中每一句的长度
        :return: output: size = (src_max_len, batch, hidden_size), 双向输出之和
        :return: hidden: size = (num_layers * num_directions, batch, hidden_size), 最后一个token的隐藏层
        '''
        embedded = self.embedding(src)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]

        outputs = self.output_dropout(outputs)

        return outputs, hidden

class LuongAttention(nn.Module):
    ## 这里考虑decoding中每一步分别执行attention，所以用来做attention的decoder_output为[1, batch, hidden_size]

    def __init__(self, method, hidden_size):
        super(LuongAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(2*hidden_size, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        '''
        :param hidden: size = (1, batch, hidden_size)
        :param encoder_output: size = (src_max_len, batch, hidden_size)
        :return: size = (src_max_len, batch), 其中第0维为source句子每一个token对应的attentionscore
        下面两个函数的参数和输出相同
        '''
        return torch.sum(hidden *  encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()  ## (batch, src_max_len)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)  ## (batch, 1, src_max_len), 后续使用需要再进行一次transpose


class TextAttnDecoderGRU(nn.Module):
    ## 不考虑图片输入的Attention Decoder. Attention采用Luong提出的方法.
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0):
        '''
        :param attn_model: is in ['dot', 'general', 'concat']
        :param embedding: 应该是目标语言
        :param output_size: 即目标语言的vocab_size
        '''
        super(TextAttnDecoderGRU, self).__init__()
        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = LuongAttention(attn_model, hidden_size)

    def forward(self, input, last_hidden, encoder_outputs):
        '''
        :param input: size = (1, batch) 某一时刻的输入
        :param last_hidden: size = (num_layers * num_directions, batch, hidden_size)
        :param encoder_outputs: size = (tgt_max_len, batch, hidden_size)
        :return output: size = (batch, output_size), prob_like
        :return hidden: size = (n_layers * num_directions, batch, hidden_size)
        '''

        embedded = self.embedding(input)
        embedded = self.embedding_dropout(embedded)
        output, hidden = self.gru(embedded, last_hidden)  ## output: size = (1, batch, hidden_size)
        attn_weights = self.attn(output, encoder_outputs)  ## size = (batch, 1, hidden_size)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))  ## size = (batch, 1, hidden)
        output = output.squeeze(0)  ## size = (batch, hidden_size)
        context = context.squeeze(1)  ## size = (batch, hidden)
        concat_input = torch.cat((output, context), 1)
        concat_output = self.concat(concat_input)
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = tgt_vocab

    def forward(self, input_seq, input_length, max_length):
        '''
        :param input_seq: size = (input_length, 1)
        :param input_length: scalar
        :param max_length: scalar
        :return: all_tokens: size = (max_length, batch)
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * tgt_vocab('b<start>')
        all_tokens = torch.zero([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        for _ in range(max_length):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores


class BeamSearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, tgt_vocab, beam_size):
        super(BeamSearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = tgt_vocab
        self.beam_size = beam_size

    def forward(self, input_seq, input_length, max_length):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * tgt_vocab('b<start>')
        # beam search的思路：一个字典保存之前的path->log概率，一个tensor存每个beam的previous hidden state,

        # prev_paths保存目前概率最高的k个path（初始只有一个）. 列表元素有两个，第一个是一个tuple保存路径（也是一个列表）以及
        # 生成下一个token所需的hidden_state
        prev_paths = [[([idx], decoder_hidden), 1.0]]
        # new_paths保存下一个token可能的情况，计算完成后将会有k**2个条目，
        new_paths = []
        hidden_size = self.hidden_size
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]

        step = 0
        while step <= max_length:
            # beam search
            for i, prev_ in enumerate(prev_paths):
                prev_condition = prev_[0]  ## 也就是之前的path和当前hidden state
                prob = prev_[1]  # 当前path概率
                prev_path, h_1, h_2 = prev_condition
                last_word_idx = prev_path[-1]
                # 如果path中的上一个token是eof，直接break
                if last_word_idx == end_vocab:
                    new_paths.append(prev_paths[i])
                    break
                # 过decoder
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_output = decoder_output.squeeze().squeeze()  # shape = (hidden_size,) 未经softmax
                assert decoder_output.shape == (hidden_size,)
                # 如果上一个词不是终止token（标点符号），则禁止输出eof
                if last_word_idx not in termination_list:
                    decoder_output[end_vocab] = -1000.0
                # 禁止输出fobidden tokens
                for forbidden in forbidden_list:
                    decoder_output[forbidden] = -1000.0

                output_prob = self.softmax(decoder_output)  # shape = (hidden_size,) prob_like
                values, indices = torch.topk(output_prob, beam_size)
                for ix in range(beam_size):
                    new_paths.append([(prev_path + [indices[ix]], decoder_hidden), prob + torch.log(values[ix] + 1e-6)])

            # 现在new_paths里应该有beam_size**2个条目
            sorted_paths = sorted(new_paths, key=itemgetter(1))
            prev_paths = sorted_paths[:beam_size]
            new_paths = []

        assert len(prev_paths) > 0
        ans = prev_paths[0][0][0]
        return ans


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
        self.softmax = nn.Softmax(0)

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

    def sample(self, text_input, image_input, maxlen, beam_size=12):
        '''
        :param text_input: shape = (batch, src_max_len, hidden_size)
        :param image_input: size = (batch, image_len, image_size)
        :return: output: size =
        论文中说用h1(1)来初始化h0(2)，但是sample的时候h1(1)还没出来就要用h0(2)了？先用相同的初始化方式了。
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vocab = self.vocab
        idx = torch.LongTensor([vocab(b'<start>')]).to(device)
        h_1 = self.init_hidden(text_input)
        h_2 = self.init_hidden(text_input)
        batch_size = text_input.shape[0]
        image_input = self.img2txt(image_input)
        text_hat, image_hat = self.text_image_attention(text_input, image_input)
        text_hat = text_hat.contiguous()
        image_hat = image_hat.contiguous()

        # beam search的思路：一个字典保存之前的path->log概率，一个tensor存每个beam的previous hidden state,

        # prev_paths保存目前概率最高的k个path（初始只有一个）. 列表元素有两个，第一个是一个tuple保存路径（也是一个列表）以及
        # 生成下一个token所需的hidden_state
        prev_paths = [[([idx], h_1, h_2), 1.0]]
        # new_paths保存下一个token可能的情况，计算完成后将会有k**2个条目，
        new_paths = []
        hidden_size = self.hidden_size
        end_vocab = vocab('<end>')
        forbidden_list = [vocab('<pad>'), vocab('<start>'), vocab('<unk>')]
        termination_list = [vocab('.'), vocab('?'), vocab('!')]
        function_list = [vocab('<end>'), vocab('.'), vocab('?'), vocab('!'), vocab('a'), vocab('an'), vocab('am'),
                         vocab('is'), vocab('was'), vocab('are'), vocab('were'), vocab('do'), vocab('does'),
                         vocab('did')]

        step = 0
        while step <= maxlen:

            # beam search
            for i, prev_ in enumerate(prev_paths):
                prev_condition = prev_[0]  ## 也就是之前的path和当前hidden state
                prob = prev_[1]  # 当前path概率
                prev_path, h_1, h_2 = prev_condition
                last_word_idx = prev_path[-1]
                # 如果path中的上一个token是eof，直接break
                if last_word_idx == end_vocab:
                    new_paths.append(prev_paths[i])
                    break
                # 过decoder
                input = self.embedding(last_word_idx).view(1,1,-1)
                decoder1_output, h_1 = self.decoder_1(input, h_1)
                text_bar, _ = self.dec_txt_attention(decoder1_output, text_hat)  # shape = (1, 1, hidden_size)
                image_bar, _ = self.dec_img_attention(decoder1_output, image_hat)
                input_2 = self.h1toh2(torch.cat([text_bar, image_bar], dim=2))
                decoder2_output, h_2 = self.decoder_2(input_2, h_2)
                decoder2_output = decoder2_output.squeeze().squeeze()  # shape = (hidden_size,) 未经softmax
                assert decoder2_output.shape == (hidden_size,)
                # 如果上一个词不是终止token（标点符号），则禁止输出eof
                if last_word_idx not in termination_list:
                    decoder2_output[end_vocab] = -1000.0
                # 禁止输出fobidden tokens
                for forbidden in forbidden_list:
                    decoder2_output[forbidden] = -1000.0

                output_prob = self.softmax(decoder2_output)  # shape = (hidden_size,) prob_like
                values, indices = torch.topk(output_prob, beam_size)
                for ix in range(beam_size):
                    new_paths.append([(prev_path+[indices[ix]], h_1, h_2), prob+torch.log(values[ix]+1e-6)])

            # 现在new_paths里应该有beam_size**2个条目
            sorted_paths = sorted(new_paths, key=itemgetter(1))
            prev_paths = sorted_paths[:beam_size]
            new_paths = []

        assert len(prev_paths) > 0
        ans = prev_paths[0][0][0]
        return ans
