import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torch.manual_seed(1000)

class LSTMTrigger(nn.Module):
    def __init__(self, model_params, args):
        super(LSTMTrigger, self).__init__()
        vocab_size, tagset_size, embedding_dim, random_dim, pretrain_embedding = model_params
        gpu = args.gpu

        self.random_embed = False
        self.lstm_hidden_dim = args.hidden_dim
        self.bilstm_flag = args.bilstm
        self.lstm_layer = args.num_layers
        self.batch_size = args.batch_size
        self.batch_mode = True if args.batch_size>1 else False

        self.use_position = args.use_position
        self.position_size = 100
        self.position_dim = 5

        # conv layer
        self.use_conv = args.use_conv
        self.in_channels = embedding_dim
        if self.use_position:
            self.in_channels += self.position_dim
        self.out_channels = args.conv_filter_num
        self.kernal_size1 = args.conv_width1
        self.kernal_size2 = args.conv_width2

        self.hidden_dim_fst = self.lstm_hidden_dim
        if self.bilstm_flag: self.hidden_dim_fst *= 2
        if self.use_conv: self.hidden_dim_fst += self.out_channels*2

        if args.hidden_dim_snd == -1: self.hidden_dim_snd = -1  # we do not use hidden linear layer
        elif args.hidden_dim_snd == 0: self.hidden_dim_snd = self.hidden_dim_fst
        else: self.hidden_dim_snd = args.hidden_dim_snd

        if random_dim >= 50: # use (append) random embedding
            self.word_embeddings = nn.Embedding(args.vocab_size, random_dim)
            self.pretrain_word_embeddings = torch.from_numpy(pretrain_embedding)
            self.random_embed = True
            embedding_dim += random_dim
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))
        self.position_embeddings = nn.Embedding(self.position_size, self.position_dim)

        self.conv1 = None
        self.conv2 = None
        self.maxp1 = None
        self.maxp2 = None
        if self.use_conv:
            self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size1)
            self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size2)
            if self.batch_mode:
                self.maxp1 = nn.MaxPool1d(self.position_size-self.kernal_size1+1)
                self.maxp2 = nn.MaxPool1d(self.position_size-self.kernal_size2+1)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(embedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer, bidirectional=self.bilstm_flag)
        if self.hidden_dim_snd != -1:
            self.fst_hidden = nn.Linear(self.hidden_dim_fst, self.hidden_dim_snd)
            self.hidden2tag = nn.Linear(self.hidden_dim_snd, tagset_size)
            self.hidden2tag_iden = nn.Linear(self.hidden_dim_snd, 2)
        else: # if hidden_dim_snd == -1, we do not use this linear layer
            self.hidden2tag = nn.Linear(self.hidden_dim_fst, tagset_size)
            self.hidden2tag_iden = nn.Linear(self.hidden_dim_fst, 2)

        if gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.position_embeddings = self.position_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            if self.use_conv: self.conv1 = self.conv1.cuda()
            if self.use_conv: self.conv2 = self.conv2.cuda()
            if self.hidden_dim_snd != -1: self.fst_hidden = self.fst_hidden.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.hidden2tag_iden = self.hidden2tag_iden.cuda()

        self.hidden = self.init_hidden(gpu)

    # init hidden of lstm
    def init_hidden(self, gpu):
        dims = (self.lstm_layer, self.batch_size, self.lstm_hidden_dim)
        if self.bilstm_flag:
            dims = (2*self.lstm_layer, self.batch_size, self.lstm_hidden_dim)
        init_value = torch.Tensor(np.random.uniform(-0.01, 0.01, dims))
        #init_value = torch.zeros(dims)
        h0 = autograd.Variable(init_value)
        c0 = autograd.Variable(init_value)
        if gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0,c0)

    # from: Variable of batch_size*sent_length*embedding_dim
    # to: Variable of batch_size*embedding_dim*sent_length
    def lstmformat2cnn(self, inputs):
        inputs = inputs.transpose(1, 2) # batch_size*embedding_dim*sent_length
        return inputs

    # from: batch_size*out_channels*1
    # to: 1*batch_size*out_channels
    def cnnformat2lstm(self, outputs):
        outputs = outputs.transpose(1, 2).transpose(0, 1) # 1*batch_size*out_channels
        return outputs

    def position_fea_in_sent(self, sent_length):
        positions = [[abs(j) for j in range(-i, sent_length-i)] for i in range(sent_length)]
        positions = [torch.LongTensor(position) for position in positions]
        #if self.batch_mode:
        #    positions = [torch.cat([position]*self.batch_size).resize_(self.batch_size, position.size(0)) for position in positions]
        positions = [torch.cat([position]*self.batch_size).resize_(self.batch_size, position.size(0)) for position in positions]
        positions = [autograd.Variable(position, requires_grad=False) for position in positions]
        return positions
        
    # batch shape: (batch_size, sent_length)
    def forward(self, batch, batch_sent_lens, gpu, debug=False, use_mask=True, is_test_flag=False):
        debug = False
        use_mask = False
        if use_mask:
            sent_length = max(batch_sent_lens.numpy())
        else:
            sent_length = batch.size(1)

        #print "## sent lens", sent_length
        positions = self.position_fea_in_sent(sent_length)
        embeds = self.word_embeddings(batch) # size: batch_size*sent_length*word_embed_size
        if debug:
            print "## sent embeds", batch.data.size(), embeds.data.size(), torch.sum(embeds.data)
            print embeds
            print "## word embedding:", type(self.word_embeddings.weight.data), self.word_embeddings.weight.data.size()
            print torch.sum(self.word_embeddings.weight.data), self.word_embeddings.weight.data#[:5, :5]
            if self.use_position:
                print "## position embedding:", self.position_embeddings.weight.requires_grad, type(self.position_embeddings.weight), type(self.position_embeddings.weight.data), self.position_embeddings.weight.data.size()
                #print self.position_embeddings.weight.data[:5]

        # output grad
        grad_debug = True
        #if debug and grad_debug:
        #    if self.word_embeddings.weight.grad is not None:
        #        print "## word embedding grad:", self.word_embeddings.weight.grad#[:5, :5]
        #    if self.position_embeddings.weight.grad is not None:
        #        print "## position embedding grad:", self.position_embeddings.weight.grad[:5]

        if self.random_embed:
            pretrain_embeds = self.pretrain_word_embeddings(batch)
            embeds = torch.cat((pretrain_embeds, embeds), -1)
        if not is_test_flag: embeds = self.drop(embeds)
        if debug:
            print "## sent embeds aft drop", embeds.data.size(), torch.sum(embeds.data)
            print embeds

# conv forward
        if self.use_conv:
            c1_embed = None
            c2_embed = None
            if not self.batch_mode:
                self.maxp1 = nn.MaxPool1d(sent_length-self.kernal_size1+1)
                self.maxp2 = nn.MaxPool1d(sent_length-self.kernal_size2+1)

            for word_id, position in enumerate(positions):
                if debug and word_id == 0:
                    print "## -------------- word_id", word_id
                if gpu: position = position.cuda()
                pos_embeds = self.position_embeddings(position)
                comb_embeds = torch.cat((embeds, pos_embeds), -1)
                #if debug and word_id == 0:
                #    print "## position", type(position.data), position.data.size()
                #    print "## pos_embeds", type(pos_embeds.data), pos_embeds.data.size()
                #    print "## comb_embeds", type(comb_embeds.data), comb_embeds.data.size()
                if self.use_position:
                    inputs = self.lstmformat2cnn(comb_embeds)
                else:
                    inputs = self.lstmformat2cnn(embeds)
                if debug and word_id == 0:
                    print "## input:", type(inputs.data), inputs.data.size()
                    print inputs.data

                c1 = self.conv1(inputs) # batch_size*out_channels*(sent_length-conv_width+1)
                p1 = self.maxp1(c1) # batch_size * out_channels * 1
                c2 = self.conv2(inputs)
                p2 = self.maxp2(c2)

                c1_embed_temp = self.cnnformat2lstm(p1)
                c2_embed_temp = self.cnnformat2lstm(p2)
                #if debug and word_id == 0:
                #    print "## c1_embed_temp:", type(c1_embed_temp.data), c1_embed_temp.data.size()
                #    print "## c2_embed_temp:", type(c2_embed_temp.data), c2_embed_temp.data.size()
                if word_id == 0:
                    c1_embed = c1_embed_temp
                    c2_embed = c2_embed_temp
                else:
                    c1_embed = torch.cat((c1_embed, c1_embed_temp), 0)
                    c2_embed = torch.cat((c2_embed, c2_embed_temp), 0)
            #if debug:
            #    print "## c1_embed:", type(c1_embed.data), c1_embed.data.size()
            #    #print c1_embed.data[:5, :5]
            #    print "## c2_embed:", type(c2_embed.data), c2_embed.data.size()
            #    #print c2_embed.data[:5, :5]

        #self.lstm.flatten_parameters()
        # lstm_out: sent_length * batch_size * hidden_dim
        embeds = embeds.transpose(0, 1)
        if not use_mask:
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
            hidden_in = lstm_out
        else:
            embeds_pack = pack_padded_sequence(embeds, batch_sent_lens.numpy())
            lstm_out, self.hidden = self.lstm(embeds_pack, self.hidden)
            hidden_in, _ = pad_packed_sequence(lstm_out)

        if self.use_conv:
            hidden_in = torch.cat((hidden_in, c1_embed, c2_embed), -1)

        hidden_in = hidden_in.transpose(0, 1).contiguous() # batch_size * sent_length * hidden_dim
        hidden_in = hidden_in.view(self.batch_size*sent_length, -1)
        if debug:
            print "## hidden in:", hidden_in.data.size(), type(hidden_in.data), hidden_in.view(self.batch_size*sent_length, -1).size()
            print hidden_in
            print hidden_in.view(self.batch_size*sent_length, -1).data

        if self.hidden_dim_snd != -1: 
            hidden_snd = self.fst_hidden(hidden_in)
            hidden_snd = F.relu(hidden_snd)
            tag_space = self.hidden2tag(hidden_snd)
            tag_space_iden = self.hidden2tag_iden(hidden_snd)
        else:
            tag_space = self.hidden2tag(hidden_in)
            tag_space_iden = self.hidden2tag_iden(hidden_in)
        tag_scores = F.log_softmax(tag_space)
        tag_scores_iden = F.log_softmax(tag_space_iden)
        return tag_space, tag_scores, tag_space_iden, tag_scores_iden
