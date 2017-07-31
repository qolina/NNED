import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMTrigger(nn.Module):
    def __init__(self, pretrain_embedding, pretrain_embed_dim, hidden_dim, vocab_size, tagset_size, dropout, bilstm, num_layers, random_dim, gpu):
        super(LSTMTrigger, self).__init__()

        embedding_dim = pretrain_embed_dim
        self.hidden_dim = hidden_dim
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

        self.drop = nn.Dropout(dropout)
        self.bilstm_flag = bilstm
        self.lstm_layer = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.lstm_layer, bidirectional=self.bilstm_flag)
        if self.bilstm_flag:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            self.hidden2tag = self.hidden2tag.cuda()

        self.hidden = self.init_hidden(gpu)

    def init_hidden(self, gpu):
        if self.bilstm_flag:
            h0 = autograd.Variable(torch.zeros(2*self.lstm_layer, 1, self.hidden_dim))
            c0 = autograd.Variable(torch.zeros(2*self.lstm_layer, 1, self.hidden_dim))
        else:
            h0 = autograd.Variable(torch.zeros(self.lstm_layer, 1, self.hidden_dim))
            c0 = autograd.Variable(torch.zeros(self.lstm_layer, 1, self.hidden_dim))
        
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0,c0)

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

        embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


