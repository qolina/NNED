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
        position_size = 80
        position_dim = 5

        # conv layer
        self.use_conv = args.use_conv
        self.in_channels = embedding_dim
        if self.use_position:
            self.in_channels += position_dim
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
        self.position_embeddings = nn.Embedding(self.position_size, position_dim)

        self.conv1 = None
        self.conv2 = None
        self.maxp1 = None
        self.maxp2 = None
        if self.use_conv:
            self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size1)
            self.conv2 = nn.Conv1d(self.in_channels, self.out_channels, self.kernal_size2)
        self.drop = nn.Dropout(args.dropout)
        self.lstm = nn.LSTM(embedding_dim, self.lstm_hidden_dim, num_layers=self.lstm_layer, bidirectional=self.bilstm_flag)

        # attention part
        self.use_attention = False
        self.concat_att = False
        self.att_lin = nn.Linear(self.lstm_hidden_dim*2, self.lstm_hidden_dim*2)
        self.att_global = nn.Linear(self.lstm_hidden_dim*2, 1, bias=False)
        self.att_multi = nn.Linear(self.lstm_hidden_dim*2, 80, bias=False)
        self.tanh = nn.Tanh()
        self.register_buffer("mask",torch.FloatTensor())


        if self.hidden_dim_snd != -1:
            self.fst_hidden = nn.Linear(self.hidden_dim_fst, self.hidden_dim_snd)
            self.hidden2tag = nn.Linear(self.hidden_dim_snd, tagset_size)
            self.hidden2tag_iden = nn.Linear(self.hidden_dim_snd, 2)
        else: # if hidden_dim_snd == -1, we do not use this linear layer
            if self.concat_att: self.hidden_dim_fst *= 2
            self.hidden2tag = nn.Linear(self.hidden_dim_fst, tagset_size)
            self.hidden2tag_iden = nn.Linear(self.hidden_dim_fst, 2)
        self.hidden = self.init_hidden(gpu)

        if gpu:
            self.drop = self.drop.cuda()
            self.word_embeddings = self.word_embeddings.cuda()
            self.position_embeddings = self.position_embeddings.cuda()
            self.lstm = self.lstm.cuda()
            if self.use_conv:
                self.conv1 = self.conv1.cuda()
                self.conv2 = self.conv2.cuda()
            if self.hidden_dim_snd != -1: self.fst_hidden = self.fst_hidden.cuda()
            self.hidden2tag = self.hidden2tag.cuda()
            self.hidden2tag_iden = self.hidden2tag_iden.cuda()
            if self.use_attention:
                self.att_lin = self.att_lin.cuda()
                self.att_global = self.att_global.cuda()
                self.att_multi = self.att_multi.cuda()
                self.tanh = self.tanh.cuda()


    # init hidden of lstm
    def init_hidden(self, gpu, last_batch_size=None):
        if last_batch_size is None: lstm_hidden_batch_size = self.batch_size
        else: lstm_hidden_batch_size = last_batch_size
        dims = (self.lstm_layer, lstm_hidden_batch_size, self.lstm_hidden_dim)
        if self.bilstm_flag:
            dims = (2*self.lstm_layer, lstm_hidden_batch_size, self.lstm_hidden_dim)
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

    def position_fea_in_sent(self, batch_size, sent_length):
        positions = [[abs(j) for j in range(-i, sent_length-i)] for i in range(sent_length)]
        positions = [torch.LongTensor(position) for position in positions]
        positions = [torch.cat([position]*batch_size).resize_(batch_size, position.size(0)) for position in positions]
        positions = [autograd.Variable(position, requires_grad=False) for position in positions]
        return positions

    # embed_sents: sent_len * batch_size * dim
    def lstm_out_global_attention(self, embed_sents, len_s, debug=False):
        embed_hidden = self.tanh(self.att_lin(embed_sents.view(embed_sents.size(0)*embed_sents.size(1), -1)))
        if debug: print "## embed_hidden", embed_hidden.size()
        attend = self.att_global(embed_hidden).view(embed_sents.size(0), embed_sents.size(1)).transpose(0, 1)
        if debug: print "## attend", attend.size()
        byte_mask = self._list_to_bytemask(list(len_s))
        if debug: print "## byte mask", byte_mask.size()
        all_att = self._masked_softmax(attend, byte_mask).transpose(0,1) # attW,sent
        if debug: print "## all_att", all_att.size()
        if debug: print "## embed_sents", embed_sents.size()
        attended = all_att.unsqueeze(2).expand_as(embed_sents) * embed_sents
        if debug: print "## attended", attended.size()
        #return attended.sum(0, True).squeeze(0), all_att
        return None, all_att

    def lstm_out_attention(self, embed_sents, len_s, debug=False):
        embed_hidden = self.tanh(self.att_lin(embed_sents.view(embed_sents.size(0)*embed_sents.size(1), -1)))
        if debug: print "## embed_hidden", embed_hidden.size()
        attend = self.att_multi(embed_hidden).view(embed_sents.size(0), embed_sents.size(1), -1).transpose(0, 1)
        if debug: print "## attend", attend.size()
        byte_mask = self._list_to_bytemask_2(list(len_s), attend.size(2))
        if debug: print "## byte mask", byte_mask.size()
        all_att = self._masked_softmax_2(attend, byte_mask).transpose(0, 1) # attW,sent
        if debug: print "## all_att", all_att.size()
        if debug: print "## embed_sents", embed_sents.size()
        embed_sents_new = embed_sents.unsqueeze(3).expand(embed_sents.size(0), embed_sents.size(1), embed_sents.size(2), attend.size(2))
        if debug: print "## new embed_sents", embed_sents_new.size()
        all_att_new = all_att.unsqueeze(3).expand(all_att.size(0), all_att.size(1), all_att.size(2), embed_sents.size(2)).transpose(2, 3)
        if debug: print "## new all_att", all_att_new.size()
        attended = all_att_new * embed_sents_new
        attended = attended.sum(3, True).squeeze(3)
        if debug: print "## attended", attended.size()

        return attended, all_att.transpose(0, 1)

    def _list_to_bytemask(self,l):
        mask = self._buffers['mask'].resize_(len(l),l[0]).fill_(1)
        for i,j in enumerate(l):
            if j != l[0]: mask[i,j:l[0]] = 0
        return mask

    def _list_to_bytemask_2(self,l, max_sent_len):
        mask = self._buffers['mask'].resize_(len(l),l[0], max_sent_len).fill_(1)
        for i,j in enumerate(l):
            if j != l[0]: 
                mask[i,j:l[0],:] = 0
                mask[i,:,j:80] = 0
            else: mask[i,:, j:80] = 0
        return mask

    def _masked_softmax(self,mat,mask):
        exp = torch.exp(mat) * Variable(mask,requires_grad=False).cuda()
        sum_exp = exp.sum(1,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    def _masked_softmax_2(self,mat,mask):
        exp = torch.exp(mat) * Variable(mask,requires_grad=False).cuda()
        sum_exp = exp.sum(2,True)+0.0001
        return exp/sum_exp.expand_as(exp)

    # batch shape: (batch_size, sent_length)
    def forward(self, batch, batch_sent_lens, gpu, debug=False, use_mask=True, is_test_flag=False, last_batch_size=None, data_flag=None):

        if last_batch_size is None: forward_batch_size = self.batch_size
        else: forward_batch_size = last_batch_size
        debug = False
        if use_mask:
            sent_length = max(batch_sent_lens.numpy())
        else:
            sent_length = batch.size(1)

        positions = self.position_fea_in_sent(forward_batch_size, sent_length)
        embeds = self.word_embeddings(batch) # size: batch_size*sent_length*word_embed_size
        if debug:
            print "## sent embeds", batch.data.size(), embeds.data.size(), torch.sum(embeds.data)
            print embeds
            print "## word embedding:", type(self.word_embeddings.weight.data), self.word_embeddings.weight.data.size()
            print torch.sum(self.word_embeddings.weight.data), self.word_embeddings.weight.data#[:5, :5]
            if self.use_position:
                print "## position embedding:", self.position_embeddings.weight.requires_grad, type(self.position_embeddings.weight), type(self.position_embeddings.weight.data), self.position_embeddings.weight.data.size()
                #print self.position_embeddings.weight.data[:5]


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
            #print self.lstm
            #print "## self hidden", self.hidden[0].data.size(), self.hidden[1].data.size()
            #print embeds_pack.data.size()
            embeds_pack = pack_padded_sequence(embeds, batch_sent_lens.numpy())
            lstm_out, self.hidden = self.lstm(embeds_pack, self.hidden)
            hidden_in, len_batch = pad_packed_sequence(lstm_out)
            #print "Before attend", hidden_in
            #_, att_values_global = self.lstm_out_global_attention(hidden_in, len_batch)
            if self.use_attention:
                hidden_in_att, att_values = self.lstm_out_attention(hidden_in, len_batch)
                if data_flag=="test_final": print att_values.sort(2, descending=True)[1][0]
                

        if self.use_attention:
            if self.concat_att: # concat word_embedding and attention result
                hidden_in = torch.cat((hidden_in, hidden_in_att), -1)
            else: hidden_in = hidden_in + hidden_in_att
        if self.use_conv:
            hidden_in = torch.cat((hidden_in, c1_embed, c2_embed), -1)

        hidden_in = hidden_in.transpose(0, 1).contiguous() # batch_size * sent_length * hidden_dim
        hidden_in = hidden_in.view(forward_batch_size*sent_length, -1)

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


