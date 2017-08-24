import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

torch.manual_seed(1)

class LSTMTrigger(nn.Module):
    def __init__(self, pretrain_embedding, pretrain_embed_dim, lstm_hidden_dim, vocab_size, tagset_size, dropout, bilstm, num_layers, random_dim, gpu, conv_width1=2, conv_width2=3, conv_filter_num=0, hidden_dim_snd=0):
        super(LSTMTrigger, self).__init__()

        embedding_dim = pretrain_embed_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.random_embed = False
        if random_dim >= 50:
            self.word_embeddings = nn.Embedding(vocab_size, random_dim)
            self.pretrain_word_embeddings = torch.from_numpy(pretrain_embedding)
            self.random_embed = True
            embedding_dim += random_dim
        else:
            self.word_embeddings = nn.Embedding(vocab_size, pretrain_embed_dim)
            print "## word embedding init", self.word_embeddings.weight.requires_grad, self.word_embeddings.weight.data[:5, :5]
            if pretrain_embedding is not None:
                self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))
            #print "## word embedding upd from pretrain", self.word_embeddings.weight.data[:5, :5]
            #print "## pretrain embedding", pretrain_embedding[:5, :5]

        self.drop = nn.Dropout(dropout)
        self.bilstm_flag = bilstm
        self.lstm_layer = num_layers

# conv layer
        self.cnn_flag = True
        self.position_size = 300
        self.position_dim = 5
        self.position_embeddings = nn.Embedding(self.position_size, self.position_dim)
        self.in_channels = embedding_dim + self.position_dim
        self.out_channels = conv_filter_num
        self.kernal_size1 = conv_width1
        self.kernal_size2 = conv_width2
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size1)
        self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size2)

        self.lstm = nn.LSTM(embedding_dim, lstm_hidden_dim, num_layers=self.lstm_layer, bidirectional=self.bilstm_flag)
        self.hidden_dim_fst = lstm_hidden_dim
        if self.bilstm_flag: self.hidden_dim_fst *= 2
        if self.cnn_flag: self.hidden_dim_fst += self.out_channels*2

        if hidden_dim_snd == 0:
            self.hidden_dim_snd = self.hidden_dim_fst
        else:
            self.hidden_dim_snd = hidden_dim_snd

        self.fst_hidden = nn.Linear(self.hidden_dim_fst, self.hidden_dim_snd)
        self.hidden2tag = nn.Linear(self.hidden_dim_snd, tagset_size)
        self.hidden2tag_iden = nn.Linear(self.hidden_dim_snd, 2)
        if gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.position_embeddings = self.position_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.conv1 = self.conv1.cuda()
            self.conv2 = self.conv2.cuda()
            self.fst_hidden = self.fst_hidden.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.hidden2tag_iden = self.hidden2tag_iden.cuda()

        self.hidden = self.init_hidden(gpu)

    def init_hidden(self, gpu):
        if self.bilstm_flag:
            dims = (2*self.lstm_layer, 1, self.lstm_hidden_dim)
        else:
            dims = (self.lstm_layer, 1, self.lstm_hidden_dim)
        init_value = np.random.uniform(-0.01, 0.01, dims)
        h0 = autograd.Variable(torch.Tensor(init_value))
        c0 = autograd.Variable(torch.Tensor(init_value))

        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0,c0)

    # from: Variable of sent_length*embedding_dim
    # to: Variable of batch_size*embedding_dim*sent_length
    def lstmformat2cnn(self, inputs):
        sent_length = inputs.size()[0]
        batch_size = 1
        inputs = inputs.view(sent_length, batch_size, -1) # sent_length*batch_size*embedding_dim
        inputs = inputs.transpose(0, 1).transpose(1, 2) # batch_size*embedding_dim*sent_length
        return inputs

    # from: batch_size*out_channels*1
    # to: 1*out_channels
    def cnnformat2lstm(self, outputs):
        outputs = outputs.transpose(1, 2).transpose(0, 1) # 1*batch_size*out_channels
        outputs = outputs.view(1, self.out_channels)
        return outputs

    def prep_position(self, sentence):
        positions_arr = [[abs(j) for j in range(-i, len(sentence)-i)] for i in range(len(sentence))]
        positions = [autograd.Variable(torch.LongTensor(position), requires_grad=False) for position in positions_arr]
        return positions
        
    def forward(self, sentence, gpu, debug=False):
        self.hidden = self.init_hidden(gpu)

        embeds = self.word_embeddings(sentence)
        positions = self.prep_position(sentence)
        if debug:
            print "## word embedding:", type(self.word_embeddings.weight.data), self.word_embeddings.weight.data.size()
            print self.word_embeddings.weight.data[:5, :5]
            print type(self.word_embeddings.weight)
            print "## position embedding:", self.position_embeddings.weight.requires_grad, type(self.position_embeddings.weight), type(self.position_embeddings.weight.data), self.position_embeddings.weight.data.size()
            print self.position_embeddings.weight.data[:5]
            print "## embeds", embeds.requires_grad, embeds.data[:10]

        #if self.word_embeddings.weight.grad is not None:
        #    print "## word embedding grad:", self.word_embeddings.weight.grad#[:5, :5]
        #if self.position_embeddings.weight.grad is not None:
        #    print "## position embedding grad:", self.position_embeddings.weight.grad[:5]
        #if embeds.grad is not None:
        #    print "## sent word embedding grad:", embeds.grad[:5, :5]
        if self.random_embed:
            pretrain_embeds = self.pretrain_word_embeddings(sentence)
            embeds = torch.cat((pretrain_embeds, embeds), 1)
            #print embeds

# conv forward
        if self.cnn_flag:
            c1_embed = None
            c2_embed = None
            self.maxp1 = nn.MaxPool1d(len(sentence)-self.kernal_size1+1)
            self.maxp2 = nn.MaxPool1d(len(sentence)-self.kernal_size2+1)

            for word_id, position in enumerate(positions):
                if debug and word_id == 0:
                    print "## -------------- word_id", word_id
                    print position.data.view(1, -1)
                if gpu: position = position.cuda()
                pos_embeds = self.position_embeddings(position)
                comb_embeds = torch.cat((embeds, pos_embeds), 1)
                inputs = self.lstmformat2cnn(comb_embeds)
                if debug and word_id == 0:
                    print "## maxp1:", type(self.maxp1)
                    print "## maxp2:", type(self.maxp2)
                    print "## input:", type(inputs.data), inputs.data.size()
                    print "## pos_embeds:", type(pos_embeds.data), pos_embeds.data.size()
                    print pos_embeds.data[:5]

                c1 = self.conv1(inputs) # batch_size*out_channels*(sent_length-conv_width+1)
                if debug and word_id == 0:
                    print "## c1:", type(c1.data), c1.data.size()
                p1 = self.maxp1(c1) # batch_size * out_channels * 1
                if debug and word_id == 0:
                    print "## p1:", type(p1.data), p1.data.size()

                c2 = self.conv2(inputs)
                if debug and word_id == 0:
                    print "## c2:", type(c2.data), c2.data.size()
                p2 = self.maxp2(c2)
                if debug and word_id == 0:
                    print "## p2:", type(p2.data), p2.data.size()

                c1_embed_temp = self.cnnformat2lstm(p1)
                c2_embed_temp = self.cnnformat2lstm(p2)
                if debug and word_id == 0:
                    print "## c1_embed_temp:", type(c1_embed_temp.data), c1_embed_temp.data.size()
                    print "## c2_embed_temp:", type(c2_embed_temp.data), c2_embed_temp.data.size()
                if word_id == 0:
                    c1_embed = c1_embed_temp
                    c2_embed = c2_embed_temp
                else:
                    c1_embed = torch.cat((c1_embed, c1_embed_temp), 0)
                    c2_embed = torch.cat((c2_embed, c2_embed_temp), 0)
            if debug:
                print "## c1_embed:", type(c1_embed.data), c1_embed.data.size()
                print c1_embed.data[:5, :5]
                print "## c2_embed:", type(c2_embed.data), c2_embed.data.size()
                print c2_embed.data[:5, :5]

        embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        if debug:
            print "## lstm out:", lstm_out.data[:10, :10]
        hidden_in = lstm_out
        if self.cnn_flag:
            #c1_embed= c1_embed.expand(len(sentence), c1_embed.size()[1])
            #c2_embed= c2_embed.expand(len(sentence), c2_embed.size()[1])
            hidden_in = torch.cat((lstm_out, c1_embed, c2_embed), 1)

        hidden_snd = self.fst_hidden(hidden_in)
        hidden_snd = F.relu(hidden_snd)
        tag_space = self.hidden2tag(hidden_snd)
        tag_scores = F.log_softmax(tag_space)
        #tag_scores = F.softmax(tag_space)
        tag_space_iden = self.hidden2tag_iden(hidden_snd)
        return tag_space, tag_scores, tag_space_iden
