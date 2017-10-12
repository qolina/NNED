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
from util import outputPRF, outputParameters
from util import output_normal_pretrain, output_dynet_format
from util import check_dataloader
from util import get_trigger, evalPRF, evalPRF_iden
from util import load_data, load_data2
from args import get_args

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as torch_data

from ace_event_dataset import MyDataset_batch, MyDataset
from lstm_trigger import LSTMTrigger
torch.manual_seed(1)
Tab = "\t"

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else len(to_ix)-1 for w in seq]
    tensor = autograd.Variable(torch.LongTensor(idxs), requires_grad=False)
    return tensor

def arr2tensor(arr):
    return torch.LongTensor(arr)

def tensor2var(eg_tensor):
    return autograd.Variable(eg_tensor, requires_grad=False)

def eval_model(data_loader, model, loss_function, data_flag, gpu):
    debug = True
    loss_all = 0
    gold_results = []
    pred_results = []
    #for sent, tags, gold_triggers in data[:]:
    for iteration, batch in enumerate(data_loader):
        sentence_in, targets = batch
        iden_targets = torch.gt(targets, torch.zeros(targets.size()).type_as(targets)) 

        sentence_in = tensor2var(sentence_in)
        targets = tensor2var(targets)
        iden_targets = tensor2var(iden_targets)

        if gpu:
            sentence_in = sentence_in.cuda()

        tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, gpu)

        _, tag_outputs = tag_scores.data.max(1)
        if gpu: tag_outputs = tag_outputs.cpu()
        if gpu: tag_scores = tag_scores.cpu()
        if gpu: tag_scores_iden = tag_scores_iden.cpu()

        for target_doc in targets:
            gold_triggers = get_trigger(target_doc.data.numpy().tolist())
            gold_results.append(gold_triggers)
            #print "eval target doc", target_doc.data.numpy().tolist()
            #print gold_triggers
        for out_doc in tag_outputs.view(args.batch_size, -1):
            sys_triggers = get_trigger(out_doc.numpy().tolist())
            pred_results.append(sys_triggers)
            #print " eval out doc", out_doc.numpy().tolist()
            #print sys_triggers

        loss = loss_function(tag_scores, targets.view(-1))
        loss += loss_function(tag_scores_iden, iden_targets.type_as(targets).view(-1))
        loss_all += loss.data[0]
    if debug:
        print "## Results for eval, sample 20"
        for i in range(10):
            print i, gold_results[i]
            print i, pred_results[i]
    prf = evalPRF(gold_results, pred_results, data_flag)
    prf_iden = evalPRF_iden(gold_results, pred_results)
    return loss_all, prf, prf_iden

def init_embedding(dim1, dim2):
    init_embedding = np.random.uniform(-0.01, 0.01, (dim1, dim2))
    return np.matrix(init_embedding)

def train_func(para_arr, args, data_sets, debug=False):

    vocab_size, tagset_size, embedding_dim = para_arr[:3]
    training_size, dev_size, test_size = para_arr[3:6]
    model_path = para_arr[6]
    gpu = args.gpu

    #training_data, dev_data, test_data, pretrain_embedding = data_sets # non-batch mode
    train_loader, dev_loader, test_loader, pretrain_embedding = data_sets

    random_dim = 10

# init model
    if not args.use_pretrain:
        pretrain_embedding = init_embedding(vocab_size, embedding_dim)

    model_params_to_feed = [vocab_size, tagset_size, embedding_dim, random_dim, pretrain_embedding]

    model = LSTMTrigger(model_params_to_feed, args)

    if args.loss_flag == "cross-entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.NLLLoss()

    parameters = filter(lambda a:a.requires_grad, model.parameters())
    if args.opti_flag == "ada":
        optimizer = optim.Adadelta(parameters, lr=args.lr)
    elif args.opti_flag == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

    if 0:
       check_dataloader(train_loader)
       #return
# training
    best_f1 = -1.0
    for epoch in range(args.epoch_num):
        training_id = 0

        for iteration, batch in enumerate(train_loader):
            model.zero_grad()
            model.hidden = model.init_hidden(gpu)
            sentence_in, targets = batch
            #print iteration, targets.numpy().tolist()
            iden_targets = torch.gt(targets, torch.zeros(targets.size()).type_as(targets)) 
            #print "--", targets.numpy().tolist()

            #if iteration == 10: break
            debug = True
            #debug = True if training_id in [0, 8, 9] else False
            #if debug and model.word_embeddings.weight.grad is not None:
            #    print "## gold tag", training_id, batch[1].numpy().tolist()
                #print "## word embedding grad:", training_id, torch.sum(model.word_embeddings.weight.grad), model.word_embeddings.weight.grad#[:5, :5]
            #if debug:
            #    print "----------- sentences", sentence_in.size()
            #    print sentence_in.numpy().tolist()
            #    print "----------- targets", targets.size()
            #    print targets.numpy().tolist()
            #    print "----------- iden targets", iden_targets.size()
            #    #print iden_targets

            sentence_in = tensor2var(sentence_in)
            targets = tensor2var(targets)
            iden_targets = tensor2var(iden_targets).type_as(targets)
            if gpu:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
                iden_targets = iden_targets.cuda()

            tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, gpu, debug)
            #if debug:
                #print "##size of tag_scores, targets, tag_scores_iden, iden_targets", tag_scores.size(), targets.view(-1).size(), tag_scores_iden.size(), iden_targets.view(-1).size()
                #print "##data of tag_scores, targets, tag_scores_iden, iden_targets"
                #print tag_scores.data
                #print targets.cpu().view(-1).data.numpy().tolist
                #print tag_scores_iden.data
                #print iden_targets.data
                #print "## eval outputs", tag_scores.data.max(1)[1]
            #if training_id in [8, 9]:
            #if debug:
            #    print "--", targets.cpu().data.numpy().tolist()
            #    for target_doc in targets.cpu().data.numpy().tolist():
            #        print "eval target doc", target_doc#.data.numpy().tolist()
            #        gold_triggers = get_trigger(target_doc)#.data.numpy().tolist())
            #        print gold_triggers

            loss = loss_function(tag_scores, targets.view(-1))
            loss += loss_function(tag_scores_iden, iden_targets.view(-1))
            loss.backward()
            optimizer.step()
            training_id += sentence_in.size(0)
            if args.batch_size>=1 and iteration % 100 == 0:
                print "## training id in batch", iteration, " is :", training_id, time.asctime()

        loss_train, prf_train, prf_train_iden = eval_model(train_loader, model, loss_function, "train", gpu)
        print "## train results on epoch:", epoch, Tab, loss_train, time.asctime(), Tab,
        outputPRF(prf_train)
        print "## Iden result:", 
        outputPRF(prf_train_iden)

# result on dev
        loss_dev, prf_dev, prf_dev_iden = eval_model(dev_loader, model, loss_function, "dev", gpu)
        if prf_dev[2] > best_f1:
            print "##-- New best dev results on epoch", epoch, Tab, best_f1, "(old best)", Tab, loss_dev, time.asctime(), Tab,
            best_f1 = prf_dev[2]
            torch.save(model, model_path)
        else:
            print "##-- dev results on epoch", epoch, Tab, best_f1, "(best f1)", Tab, loss_dev, time.asctime(), Tab,
        outputPRF(prf_dev)
        print "## Iden result:",
        outputPRF(prf_dev_iden)
# result on test
        if epoch >= 10 and epoch % 10 == 0:
            if epoch % 100 == 0:
                model_test = torch.load(model_path)
                loss_test, prf_test, prf_test_iden = eval_model(test_loader, model_test, loss_function, "test_final", gpu)
            else:
                loss_test, prf_test, prf_test_iden = eval_model(test_loader, model, loss_function, "test", gpu)
            print "##-- test results on epoch", epoch, Tab, loss_test, time.asctime(), Tab,
            outputPRF(prf_test)
            print "## Iden result:",
            outputPRF(prf_test_iden)

# final result on test
    model = torch.load(model_path)
    loss_test, prf_test, prf_test_iden = eval_model(test_loader, model, loss_function, "test_final", gpu)
    print "## Final results on test", loss_test, time.asctime(), Tab,
    outputPRF(prf_test)
    print "## Iden result:",
    outputPRF(prf_test_iden)

if __name__ == "__main__":
    args = get_args()

    #######################
    ## default parameters
    embedding_dim = args.embed_dim

    #######################
    ## load datasets
    if len(args.dev) > 0:
        training_data, dev_data, test_data, vocab, tags_data, pretrain_embedding, model_path = load_data2(args)
    else:
        training_data, dev_data, test_data, vocab, tags_data, pretrain_embedding, model_path = load_data(args)
        if args.test_as_dev:
            dev_data = test_data
        else:
            if 1:
                random.shuffle(training_data, lambda: 0.3) # shuffle data before get dev
            training_data = training_data[:-500]
            dev_data = training_data[-500:]
    model_path = model_path + "_" + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_"

    vocab_size = len(vocab)
    pretrain_vocab_size, pretrain_embed_dim = pretrain_embedding.shape
    tagset_size = len(tags_data)

    if 0:
        all_data = training_data+dev_data+test_data
        sent_lens = [len(item[0]) for item in all_data]
        print "## Statistic sent length:", max(sent_lens), min(sent_lens)
        sys.exit(0)
    if 0:
        output_normal_pretrain(pretrain_embedding, vocab, "../ni_data/f.ace.pretrain300.vectors")
        output_dynet_format(training_data, vocab, tags_data, "../ni_data/f.ace_trigger.train")
        output_dynet_format(dev_data, vocab, tags_data, "../ni_data/f.ace_trigger.dev")
        output_dynet_format(test_data, vocab, tags_data, "../ni_data/f.ace_trigger.test")
        sys.exit(0)


    #######################
    ## store and output all parameters
    if args.use_pretrain: embedding_dim = pretrain_embed_dim

    param_str = "p"+str(embedding_dim) + "_hd" + str(args.hidden_dim) + "_2hd" + str(args.hidden_dim_snd)
    if args.use_conv: param_str += "_f" + str(args.conv_filter_num) + "_c" + str(args.conv_width1) + "_c" + str(args.conv_width2)
    if args.use_pretrain: param_str += "_pf"
    param_str += "_lr" + str(args.lr*100)# + "_" + str() + "_" + str()
    model_path += param_str

    para_arr = [vocab_size, tagset_size, embedding_dim]
    para_arr.extend([len(training_data), len(dev_data), len(test_data), model_path])
    outputParameters(para_arr, args)

    #######################
    # dataset prepare

    #### Non_batch mode
    #training_data = [(item[0], item[1], get_trigger(item[1])) for item in training_data]
    #dev_data = [(item[0], item[1], get_trigger(item[1])) for item in dev_data]
    #test_data = [(item[0], item[1], get_trigger(item[1])) for item in test_data]
    #data_sets = training_data, dev_data, test_data, pretrain_embedding

    #### Batch mode
    # max length = 297
    train_sents = [item[0] for item in training_data]
    train_labels = [item[1] for item in training_data]
    dev_sents = [item[0] for item in dev_data]
    dev_labels = [item[1] for item in dev_data]
    test_sents = [item[0] for item in test_data]
    test_labels = [item[1] for item in test_data]

    if args.batch_size > 1:
        train_dataset = MyDataset_batch(train_sents, train_labels)
        dev_dataset = MyDataset_batch(dev_sents, dev_labels)
        test_dataset = MyDataset_batch(test_sents, test_labels)
    else:
        train_dataset = MyDataset(train_sents, train_labels)
        dev_dataset = MyDataset(dev_sents, dev_labels)
        test_dataset = MyDataset(test_sents, test_labels)
        #print train_dataset.__getitem__(8)
    train_loader = torch_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train, drop_last=True)
    dev_loader = torch_data.DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train, drop_last=True)
    test_loader = torch_data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train, drop_last=True)
    data_sets = train_loader, dev_loader, test_loader, pretrain_embedding

    #check_dataloader(train_loader)
    #sys.exit(1)

    # begin to train
    train_func(para_arr, args, data_sets)

