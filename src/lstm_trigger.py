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
            if pretrain_embedding is not None:
                self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

        #self.word_embeddings.weight.requires_grad = False
        self.drop = nn.Dropout(dropout)
        self.bilstm_flag = bilstm
        self.lstm_layer = num_layers

# conv layer
        self.cnn_flag = True
        self.in_channels = embedding_dim
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
    def lstmformat2cnn(self, inputs, gpu):
        sent_length = inputs.size()[0]
        batch_size = 1
        inputs = inputs.view(sent_length, batch_size, -1) # sent_length*batch_size*embedding_dim
        inputs = inputs.transpose(0, 1).transpose(1, 2) # batch_size*embedding_dim*sent_length
        return inputs

    # from: batch_size*out_channels*1
    # to: 1*out_channels
    def cnnformat2lstm(self, outputs, gpu):
        outputs = outputs.transpose(1, 2).transpose(0, 1) # 1*batch_size*out_channels
        outputs = outputs.view(1, self.out_channels)
        return outputs

    def forward(self, sentence, gpu):
        self.hidden = self.init_hidden(gpu)

        embeds = self.word_embeddings(sentence)
        #print embeds

        if self.random_embed:
            sent_tensor = sentence.data
            embeds = embeds.data
            if gpu: sent_tensor = sent_tensor.cpu()
            if gpu: embeds = embeds.cpu()
            pretrain_embeds = torch.index_select(self.pretrain_word_embeddings, 0, sent_tensor)
            embeds = torch.cat((pretrain_embeds, embeds.double()), 1)
            embeds = Variable(embeds.float())
            if gpu: embeds = embeds.cuda()
        #print embeds
# conv forward
        if self.cnn_flag:
            inputs = self.lstmformat2cnn(embeds, gpu)
            self.maxp1 = nn.MaxPool1d(len(sentence)-self.kernal_size1+1)
            self.maxp2 = nn.MaxPool1d(len(sentence)-self.kernal_size2+1)
            c1 = self.conv1(inputs) # batch_size*out_channels*(sent_length-conv_width+1)
            p1 = self.maxp1(c1) # batch_size * out_channels * 1
            c2 = self.conv2(inputs)
            p2 = self.maxp2(c2)
            c1_embed = self.cnnformat2lstm(p1, gpu)
            c2_embed = self.cnnformat2lstm(p2, gpu)

        embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out = lstm_out.view(len(sentence), -1)
        hidden_in = lstm_out
        if self.cnn_flag:
            #c1_embed.data = c1_embed.data.expand(len(sentence), c1_embed.size()[1])
            #c2_embed.data = c2_embed.data.expand(len(sentence), c2_embed.size()[1])
            #hidden_in = torch.cat((lstm_out.data, c1_embed.data, c2_embed.data), 1)
            c1_embed= c1_embed.expand(len(sentence), c1_embed.size()[1])
            c2_embed= c2_embed.expand(len(sentence), c2_embed.size()[1])
            hidden_in = torch.cat((lstm_out, c1_embed, c2_embed), 1)
            #hidden_in = Variable(hidden_in)
            #print hidden_in
            #print type(hidden_in)

        hidden_snd = self.fst_hidden(hidden_in)
        hidden_snd = F.relu(hidden_snd)
        tag_space = self.hidden2tag(hidden_snd)
        tag_scores = F.log_softmax(tag_space)
        tag_space_iden = self.hidden2tag_iden(hidden_snd)
        return tag_space, tag_scores, tag_space_iden


