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
#from get_constituent_topdown_oracle import unkify
from util import outputPRF, loadVocab, loadTrainData, loadPretrain

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from lstm_trigger import LSTMTrigger
torch.manual_seed(1)
Tab = "\t"

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else len(to_ix)-1 for w in seq]
    tensor = autograd.Variable(torch.LongTensor(idxs), requires_grad=False)
    return tensor

def arr2tensor(arr):
    tensor = autograd.Variable(torch.LongTensor(arr), requires_grad=False)
    return tensor

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

def eval_model(data, model, loss_function, data_flag, gpu):
    loss_all = 0
    gold_results = []
    pred_results = []
    for sent, tags in data:

        sentence_in = arr2tensor(sent)
        targets = arr2tensor(tags)

        if gpu:
            sentence_in = sentence_in.cuda()
            targets = targets.cuda()

        tag_scores = model(sentence_in, gpu)
        if gpu: tag_scores = tag_scores.cpu()

        tag_outputs = tag_scores.data.numpy().argmax(axis=1)
        gold_results.append(get_trigger([t for t in tags]))
        pred_results.append(get_trigger(tag_outputs.tolist()))

        if gpu: targets = targets.cpu()
        loss = loss_function(tag_scores, targets)
        loss_all += loss.data.numpy()[0]
    prf = evalPRF(gold_results, pred_results)
    return loss_all, prf


def example(model, sent, gpu):
    example_sent_in = arr2tensor(sent)
    if gpu:
        example_sent_in = example_sent_in.cuda()
    example_model_out = model(example_sent_in)
    if gpu:
        example_model_out = example_model_out.cpu()
    example_out = example_model_out.data.numpy().argmax(axis=1)
    print "## results on example sent:", 
    print example_out

def check_data(data, vocab):
    #id2word = dict([(word_index, word) for word, word_index in vocab])

    vocab_index = [word_index for item in data for word_index in item[0]]
    counter_index = Counter(vocab_index)
    vocab_size = len(counter_index)
    word_df_one = [word_index for word_index, df in counter_index.items() if df == 1]
    #print min(vocab_index), counter_index[min(vocab_index)], max(vocab_index), counter_index[max(vocab_index)], min(counter_index.values()), max(counter_index.values())
    new_data = [([word_index if word_index not in word_df_one else vocab_size-1 for word_index in sent], tags) for sent, tags in data]
    for sent, tags in data:
        if len(sent) < 1 or len(tags) < 1:
            print "-- 0-length data", sent, tags
    return new_data

def load_data():
    train_filename, pretrain_embedding_filename, tag_filename, vocab_filename, test_filename = parseArgs(sys.argv)

# pretrain embedding: matrix (vocab_size, pretrain_embed_dim)
    pretrain_embedding = loadPretrain(pretrain_embedding_filename)
    print "## pretrained embedding loaded.", time.asctime()

# vocab: word: word_id
    vocab = loadVocab(vocab_filename)
    print "## vocab loaded.", time.asctime()

# train test
    training_data = loadTrainData(train_filename)
    print "## train loaded.", train_filename, time.asctime()
    training_data = check_data(training_data, vocab)
    test_data = loadTrainData(test_filename)
    print "## test loaded.", test_filename, time.asctime()
    test_data = check_data(test_data, vocab)

# tags_data: tag_name: tag_id
    tags_data = loadVocab(tag_filename)
    tags_data["NULL"] = 0
    print "## event tags loaded.", time.asctime()

    return training_data, test_data, vocab, tags_data, pretrain_embedding


def get_random_embedding(vocab_size, random_dim):
    random_embedding = np.random.uniform(-1, 1, (vocab_size, random_dim))
    return np.matrix(random_embedding)


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
    arg3 = getArg(args, "-tag")
    #arg4 = getArg(args, "-dev")
    arg4 = getArg(args, "-vocab")
    arg5 = getArg(args, "-test")
    return [arg1, arg2, arg3, arg4, arg5]


def main():

    training_data, test_data, vocab, tags_data, pretrain_embedding = load_data()
    training_data = training_data[:-500]
    dev_data = training_data[-500:]
    vocab_size, pretrain_embed_dim = pretrain_embedding.shape
    tagset_size = len(tags_data)
    #sys.exit(0)

    random_dim = 50

    gpu = torch.cuda.is_available()
    print "gpu available:", gpu
    #gpu = false
    dropout = 0.5
    bilstm = True
    num_layers = 1
    iteration_num = 30
    Hidden_dim = 100
    learning_rate = 0.05
    Embedding_dim = pretrain_embed_dim

    para_arr = [vocab_size, tagset_size, Embedding_dim, Hidden_dim]
    para_arr.extend([dropout, bilstm, num_layers, gpu, iteration_num, learning_rate])
    para_arr.extend([len(training_data), len(dev_data), len(test_data)])
    outputParameters(para_arr)
    #sys.exit(0)

# init model
    model = LSTMTrigger(pretrain_embedding, pretrain_embed_dim, Hidden_dim, vocab_size, tagset_size, dropout, bilstm, num_layers, random_dim, gpu)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# training
    for epoch in range(iteration_num):
        for sent, tags in training_data:

            model.zero_grad()
            model.hidden = model.init_hidden(gpu)

            sentence_in = arr2tensor(sent)
            targets = arr2tensor(tags)

            if gpu:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()

            tag_scores = model(sentence_in, gpu)

            if gpu:
                tag_scores = tag_scores.cpu()
                targets = targets.cpu()
                #print len(tag_scores), len(targets)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            #if sentence_id % 2000 == 0:
            #    print "## ", sentence_id, time.asctime()

        if epoch == 0 or (epoch+1) % 1 == 0:
            loss_train, prf_train = eval_model(training_data, model, loss_function, "train", gpu)
            print "## train results on epoch:", epoch, Tab, loss_train, time.asctime(), Tab,
            outputPRF(prf_train)
        if epoch % (3-1) == 0:
            loss_dev, prf_dev = eval_model(dev_data, model, loss_function, "dev", gpu)
            print "##-- dev results on epoch", epoch, Tab, loss_dev, time.asctime(), Tab,
            outputPRF(prf_dev)
        if epoch % 10 == 0:
            loss_test, prf_test = eval_model(test_data, model, loss_function, "test", gpu)
            print "##-- test results on epoch", epoch, Tab, loss_test, time.asctime(), Tab,
            outputPRF(prf_test)

    loss_test, prf_test = eval_model(test_data, model, loss_function, "test", gpu)
    print "## test results", loss_test, time.asctime(), Tab,
    outputPRF(prf_test)


if __name__ == "__main__":
    print "Usage: python .py -train trainFile -embed embeddingFile -ace aceArgumentFile -dev devFile -test testFile"
    print sys.argv

    main()

