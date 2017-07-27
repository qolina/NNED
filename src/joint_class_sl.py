# This file mainly implements a basic attention model for neural event extraction
# input file consists of sentences which contain one-event.
import os
import re
import sys
import time
import random
import cPickle
from collections import Counter
from aceEventUtil import loadEventHierarchy

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)
Tab = "\t"

class LSTMTrigger(nn.Module):
    def __init__(self, pretrain_embedding, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout, bilstm, num_layers, gpu):
        super(LSTMTrigger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        if pretrain_embedding is not None:
            #print pretrain_embedding
            self.word_embeddings.weight.data.copy_(torch.from_numpy(pretrain_embedding))

        if dropout != -1:
            self.drop = nn.Dropout(dropout)
        self.bilstm_flag = bilstm
        self.lstm_layer = num_layers

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=self.lstm_layer, bidirectional=self.bilstm_flag)
        if self.bilstm_flag:
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        if gpu:
            if dropout != -1:
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

    def forward(self, sentence):
        self.hidden = self.init_hidden(True)

        embeds = self.word_embeddings(sentence)
        if dropout != -1:
            embeds = self.drop(embeds)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else len(to_ix)-1 for w in seq]
    tensor = autograd.Variable(torch.LongTensor(idxs), requires_grad=False)
    return tensor

def prepare_word_map(training_data):
    word_map = {}
    for train_item in training_data:
        sent_id, sent_text, event_type_seq = train_item[:3]
        for word in sent_text.split():
            if word not in word_map:
                word_map[word] = len(word_map)
    print "## Statistics of #words", len(word_map)
    return word_map

def loadWord2Vec(modelPath):
   content = open(modelPath, "r").readlines()
   wordNum, dim = content[0].strip().split()
   wordNum = int(wordNum)
   dim = int(dim)
   content = [line.strip().split() for line in content[1:]]

   words = [item[0] for item in content]
   embeddings = [[float(val) for val in item[1:]] for item in content]

   #wordIDHash = {word:wordID}
   wordIDHash = dict(zip(words, range(wordNum)))

   # add unk
   unk_word, unk_vector = ("<unk>", [random.random() for _ in range(dim)])
   wordIDHash[unk_word] = wordNum
   embeddings.append(unk_vector)

   return len(wordIDHash), dim, wordIDHash, np.matrix(embeddings)

def outputPRF(arr):
    arr = ["%.2f"%i for i in arr]
    print "-- Pre, Rec, F1:", Tab, arr[0], Tab, arr[1], Tab, arr[2]

def get_trigger(sent_tags):
    triggers = [(word_idx, tag) for word_idx, tag in enumerate(sent_tags) if tag != 0]
    return triggers

def evalPRF(items_in_docs_gold, items_in_docs):
    debug = False
    if 0:
        print items_in_docs_gold
        print items_in_docs
    common_in_docs = []
    num_in_docs_gold = []
    num_in_docs = []
    for items_in_doc, items_in_doc_gold in zip(items_in_docs_gold, items_in_docs):
        common_in_doc = [1 for item_gold, item in zip(items_in_doc_gold, items_in_doc) if item_gold == item]

        common_in_docs.append(len(common_in_doc))
        num_in_docs_gold.append(len(items_in_doc_gold))
        num_in_docs.append(len(items_in_doc))

    common = sum(common_in_docs)
    num_gold = sum(num_in_docs_gold)
    num = sum(num_in_docs)

    if debug:
        print "-- common, num_gold, num:", common, num_gold, num
        print "-- common_in_docs", common_in_docs
        print "-- num_in_docs_gold", num_in_docs_gold
        print "-- num_in_docs", num_in_docs

    if common == 0: return 0.0, 0.0, 0.0

    pre = common*100.0/num
    rec = common*100.0/num_gold
    f1 = 2*pre*rec/(pre+rec)

    return pre, rec, f1

def eval_model(data, model, data_flag, gpu):
    loss_all = 0
    gold_results = []
    pred_results = []
    for data_item in data:
        sentence_id, sentence, tags = data_item[:3]

        sentence_in = prepare_sequence(sentence.split(), Vocab_pretrain)
        targets = prepare_sequence(tags, ACE_event_types)

        if gpu:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()

        tag_scores = model(sentence_in)
        if gpu: tag_scores = tag_scores.cpu()

        tag_outputs = tag_scores.data.numpy().argmax(axis=1)
        gold_results.append(get_trigger([ACE_event_types[t] for t in tags]))
        pred_results.append(get_trigger(tag_outputs.tolist()))

        if gpu: targets = targets.cpu()
        loss = loss_function(tag_scores, targets)
        loss_all += loss.data.numpy()[0]
    prf = evalPRF(gold_results, pred_results)
    return loss_all, prf
##############
def getArg(args, flag):
    arg = None
    if flag in args:
        arg = args[args.index(flag)+1]
    return arg

# arguments received from arguments
def parseArgs(args):
    arg1 = getArg(args, "-train")
    arg2 = getArg(args, "-embed")
    arg3 = getArg(args, "-ace")
    arg4 = getArg(args, "-dev")
    arg5 = getArg(args, "-test")
    return [arg1, arg2, arg3, arg4, arg5]


if __name__ == "__main__":
    print "Usage: python .py -train trainFile -embed embeddingFile -ace aceArgumentFile -dev devFile -test testFile"
    print sys.argv
    trainFilename, embeddingFilename, ace_hierarchy_filename, dev_filename, test_filename = parseArgs(sys.argv)

    gpu = torch.cuda.is_available()
    print "GPU available:", gpu
    #gpu = False
    dropout = 0.5
    bilstm = True
    num_layers = 1
    iteration_num = 100

    vocab_size_pretrain, Embedding_dim, Vocab_pretrain, EmbeddingArr = loadWord2Vec(embeddingFilename)
    print "## pretrained embedding loaded.", time.asctime()

# train word map
    trainFile = open(trainFilename, "r")
    training_data = cPickle.load(trainFile)
    word_map_train = prepare_word_map(training_data) # word:word_idx
    print "## training data loaded.", len(training_data), time.asctime()

# dev
    dev_file = open(dev_filename, "r")
    dev_data = cPickle.load(dev_file)
    print "## dev data loaded.", len(dev_data), time.asctime()
# test
    test_data = cPickle.load(open(test_filename, "r"))
# tags
    ACE_event_types, _ = loadEventHierarchy(ace_hierarchy_filename)
    ACE_event_types = dict([(sub_type, event_type_id+1) for event_type_id, sub_type in enumerate(sorted(ACE_event_types.keys()))])
    ACE_event_types[-1] = 0
    print "## event hierarchy loaded.", time.asctime()

    Hidden_dim = 100
    vocab_size_train = len(word_map_train)
    Embedding_dim = 100
    tagset_size = len(ACE_event_types)
    example_sent_in = prepare_sequence(training_data[0][1].split(), Vocab_pretrain)
    if gpu:
        example_sent_in = example_sent_in.cuda()


# init model
    model = LSTMTrigger(EmbeddingArr, Embedding_dim, Hidden_dim, vocab_size_pretrain, tagset_size, dropout, bilstm, num_layers, gpu)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    example_model_out = model(example_sent_in)
    if gpu:
        example_model_out = example_model_out.cpu()
    example_out = example_model_out.data.numpy().argmax(axis=1)
    print "## before training, results on example sent:", 
    print example_out

# training
    for epoch in range(iteration_num):
        for train_item in training_data:
            sentence_id, sentence, tags = train_item[:3]

            model.zero_grad()
            model.hidden = model.init_hidden(gpu)

            sentence_in = prepare_sequence(sentence.split(), Vocab_pretrain)
            targets = prepare_sequence(tags, ACE_event_types)

            if gpu:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()

            tag_scores = model(sentence_in)

            if gpu:
                tag_scores = tag_scores.cpu()
                targets = targets.cpu()

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            #if sentence_id % 2000 == 0:
            #    print "## ", sentence_id, time.asctime()

        if epoch == 0 or (epoch+1) % 1 == 0:
            loss_train, prf_train = eval_model(training_data, model, "train", gpu)
            print "## train results on epoch:", epoch, Tab, loss_train, time.asctime(), Tab,
            outputPRF(prf_train)
        if epoch % (3-1) == 0:
            loss_dev, prf_dev = eval_model(dev_data, model, "dev", gpu)
            print "##-- dev results on epoch", epoch, Tab, loss_dev, time.asctime(), Tab,
            outputPRF(prf_dev)
        if epoch % 20 == 0:
            loss_test, prf_test = eval_model(test_data, model, "test", gpu)
            print "##-- test results on epoch", epoch, Tab, loss_test, time.asctime(), Tab,
            outputPRF(prf_test)

    example_model_out = model(example_sent_in)
    if gpu:
        example_model_out = example_model_out.cpu()
    example_out = example_model_out.data.numpy().argmax(axis=1)
    print "## after training, result on example_sent:",
    print example_out

    loss_test, prf_test = eval_model(test_data, model, "test", gpu)
    print "## test results", loss_test, time.asctime(), Tab,
    outputPRF(prf_test)
