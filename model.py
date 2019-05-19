import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab, dropout_rate, num_layers=1):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size  # 400
        self.vocab_size = len(vocab)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.GRU(embed_size, hidden_size,
                               batch_first=True, num_layers=num_layers, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)

    def get_params(self):
        return list(self.encoder.parameters())

    def forward(self, src):
        '''
        :param src: size = (batch, src_max_len), dtype=long
        :param tgt: size = (batch, tgt_max_len), dtype=long
        :return: output: size = (batch, max_len, num_direction*hidden_size)
        '''
        embed = self.embedding(src)
        output, h_n = self.encoder(embed)  # (batch, seq_len, num_layers*hidden_size)

        return output


class AttnDecoder(nn.Module):
    '''
    NAACL论文里的decoder为conditional decoder，这里先实现普通decoder
    '''
    def __init__(self, embed_size, hidden_size, vocab, input_size, dropout_rate, num_layers=1, attn='dot'):
        super(AttnDecoder, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab = vocab
        self.attn = attn
        self.vocab_size = len(vocab)
        self.decoder = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=num_layer, dropout=dropout_rate)
        self.embedding = nn.Embedding(self.vocab_size, embed_size)  # 这里也是，先不设置tying
        self.embedding.weight.requires_grad = False
        self.linear = nn.Linear(num_layers*hidden_size, self.vocab_size)
        self.compat = None
        self.concat = None
        self.imp = None
        if attn == 'general':
            self.compat = nn.Bilinear(hidden_size, input_size, input_size)
        elif attn == 'concat':
            self.concat = nn.Linear(hidden_size+input_size, imp_size)
            self.imp = nn.Linear(imp_size, 1)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(2)

    def get_params(self):
        params = list(self.decoder.parameters()) + list(self.linear.parameters())
        if self.attn == 'general':
            params += list(self.compat.parameters())
        if self.attn == 'concat':
            params += list(self.concat.parameters()) + list(self.imp.parameters())
        return params

    def forward(self, text_input, image_input, tgt):
        '''
        :param text_input: shape = (batch, text_len, 
        :param image_input:
        :param tgt:
        :return:
        '''
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        embed = self.embedding(tgt_input)
        output, (h_n, c_n) = self.decoder(embed)
        output = self.attention(input, output, self.attn)

        return output, tgt_output

    def attention(self, input, output, attn):
        '''
        :param input: size = (batch, src_len, -1)
        :param output: size = (batch, tgt_len, -1)
        :param attn: size = (batch, tgt_len, -1)
        :return:
        '''

        if attn == 'dot':
            a = torch.matmul(output, input.transpose(1,2))     # size = (batch, tgt_len, src_len)
            a = self.softmax(a)
            attended = torch.matmul(a, input)
        elif attn == 'general':
            a = self.compat(output, input)
            a = self.softmax(a)
            attended = torch.matmul(a, input)
        elif attn == 'concat':
            src_len = input.shape[1]
            tgt_len = output.shape[1]
            a = self.concat(torch.cat([output.unsqueeze(2).expand(-1,-1,src_len,-1),
                                      input.unsqueeze(1).expand(-1,tgt_len,-1,-1)], dim=3))
            a = self.softmax(self.imp(a).squeeze())
            attended = torch.matmul(a, input)
        elif attn == 'no':
            attended = output
        else:
            raise NameError('Invalid attention name')

        output = self.linear(attended)

        return output

    def sample(self, src_hidden, maxlen):
        '''
        :param src_hidden: size = (batch, src_max_len, -1)
        :return: output: size =
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vocab = self.vocab
        idx = torch.LongTensor([vocab(b'<start>')]).to(device)
        input = self.embedding(idx).view(1,1,-1)
        ans = []
        while True:
            output, (h, c) = self.decoder(input)
            output = self.attention(src_hidden, output, self.attn)
            output = output.squeeze().squeeze()
            max, id = torch.max(output, 0)
            ans.append(id)
            if vocab.idx2word[id.item()] == b'<end>':
                ans = ans[:-1]
                break
            if len(ans) >= maxlen:
                break
            input = self.embedding(id).view(1,1,-1)

        ans = torch.stack(ans, 0)
        return ans
