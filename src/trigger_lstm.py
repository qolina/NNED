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
from util import load_data, load_data2, sort_data
from util import pad_batch
from args import get_args

import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as torch_data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ace_event_dataset import MyDataset
from lstm_trigger import LSTMTrigger
torch.manual_seed(1000)
Tab = "\t"

def arr2tensor(arr):
    return torch.LongTensor(arr)

def tensor2var(eg_tensor):
    return autograd.Variable(eg_tensor, requires_grad=False)

def eval_model(data_loader, model, loss_function, data_flag, gpu, vocab=None, tags_data=None):
    debug = False
    loss_all = 0
    gold_results = []
    pred_results = []
    pred_results_iden = []
    debug_sents = []
    for iteration, batch in enumerate(data_loader):
        sentence_in, targets, batch_sent_lens = batch
        iden_targets = torch.gt(targets, torch.zeros(targets.size()).type_as(targets)) 

        if sentence_in.size(0) != args.batch_size: eval_batch_size = sentence_in.size(0)
        else: eval_batch_size = args.batch_size

        if sentence_in.size(0) != args.batch_size: model.hidden = model.init_hidden(gpu, last_batch_size=eval_batch_size)
        else: model.hidden = model.init_hidden(gpu)

        sentence_in = tensor2var(sentence_in)
        targets = tensor2var(targets)
        iden_targets = tensor2var(iden_targets).type_as(targets)

        if gpu:
            sentence_in = sentence_in.cuda()

        if sentence_in.size(0) != args.batch_size: 
            tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, batch_sent_lens, gpu, is_test_flag=True, last_batch_size=eval_batch_size)
        else:
            tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, batch_sent_lens, gpu, is_test_flag=True)

        _, tag_outputs = tag_scores.data.max(1)
        _, tag_outputs_iden = tag_scores_iden.data.max(1)
        if gpu: 
            tag_outputs = tag_outputs.cpu()
            tag_outputs_iden = tag_outputs_iden.cpu()
            tag_scores = tag_scores.cpu()
            tag_scores_iden = tag_scores_iden.cpu()
            tag_space = tag_space.cpu()
            tag_space_iden = tag_space_iden.cpu()
            sentence_in = sentence_in.cpu()


        debug_sents.extend(sentence_in.data.numpy().tolist())
        gold_targets = targets.data.numpy().tolist()
        pred_outputs = tag_outputs.view(eval_batch_size, -1).numpy().tolist()
        pred_idens = tag_outputs_iden.view(eval_batch_size, -1).numpy().tolist()
        for target_doc, out_doc, out_doc_iden in zip(gold_targets, pred_outputs, pred_idens):
            gold_triggers = get_trigger(target_doc)
            gold_results.append(gold_triggers)
            #print "eval target doc", target_doc
            #print gold_triggers
            sys_triggers = get_trigger(out_doc)
            pred_results.append(sys_triggers)
            #print " eval out doc", out_doc.numpy().tolist()
            #print sys_triggers
            sys_triggers_iden = get_trigger(out_doc_iden)
            pred_results_iden.append(sys_triggers_iden)

        if args.loss_flag == "nlloss":
            loss = loss_function(tag_scores, targets.view(-1)) + loss_function(tag_scores_iden, iden_targets.view(-1))
        elif args.loss_flag == "cross-entropy":
            loss = loss_function(tag_space, targets.view(-1)) + loss_function(tag_space_iden, iden_targets.view(-1))
        loss_all += loss.data[0]
    if debug:
        print "## Results for eval, sample 20"
        for i in range(10):
            print i, gold_results[i]
            print i, pred_results[i]
    prf = evalPRF(gold_results, pred_results, data_flag, debug_sents=debug_sents, vocab=vocab, tags_data=tags_data)
    prf_iden = evalPRF_iden(gold_results, pred_results)
    #prf_iden2 = evalPRF_iden(gold_results, pred_results_iden)
    return loss_all, prf, prf_iden
    #return loss_all, prf, (prf_iden, prf_iden2)

def init_embedding(dim1, dim2):
    init_embedding = np.random.uniform(-0.01, 0.01, (dim1, dim2))
    return np.matrix(init_embedding)

def train_func(para_arr, args, data_sets, debug=False):

    vocab_size, tagset_size, embedding_dim = para_arr[:3]
    training_size, dev_size, test_size = para_arr[3:6]
    model_path = para_arr[6]
    gpu = args.gpu

    train_loader, dev_loader, test_loader, pretrain_embedding, vocab, tags_data = data_sets

    random_dim = 10

# init model
    if not args.use_pretrain:
        pretrain_embedding = init_embedding(vocab_size, embedding_dim)

    model_params_to_feed = [vocab_size, tagset_size, embedding_dim, random_dim, pretrain_embedding]

    if 0:
       check_dataloader(train_loader, vocab_size)
       check_dataloader(dev_loader, vocab_size)
       check_dataloader(test_loader, vocab_size)
       return

    model = LSTMTrigger(model_params_to_feed, args)

    if args.loss_flag == "cross-entropy":
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function = nn.NLLLoss()

    parameters = filter(lambda a:a.requires_grad, model.parameters())
    if args.opti_flag == "adadelta":
        optimizer = optim.Adadelta(parameters, lr=args.lr)
    elif args.opti_flag == "sgd":
        #optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=1e-4)
        optimizer = optim.SGD(parameters, lr=args.lr)
    elif args.opti_flag == "adam":
        optimizer = optim.Adam(parameters, lr=args.lr)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)

# training
    best_f1 = -1.0
    best_epoch = -1
    for epoch in range(args.epoch_num):
        #scheduler.step()
        training_id = 0

        for iteration, batch in enumerate(train_loader):
            model.zero_grad()
            sentence_in, targets, batch_sent_lens = batch
            if sentence_in.size(0) != args.batch_size: model.hidden = model.init_hidden(gpu, last_batch_size=sentence_in.size(0))
            else: model.hidden = model.init_hidden(gpu)
            #print "## model hidden", model.hidden[0].data.size(), model.hidden[1].data.size()

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

            #print sentence_in, targets, batch_sent_lens
            #continue
            sentence_in = tensor2var(sentence_in)
            targets = tensor2var(targets)
            iden_targets = tensor2var(iden_targets).type_as(targets)

            if gpu:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
                iden_targets = iden_targets.cuda()

            if sentence_in.size(0) != args.batch_size: 
                tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, batch_sent_lens, gpu, debug=debug, last_batch_size=sentence_in.size(0))
            else:
                tag_space, tag_scores, tag_space_iden, tag_scores_iden = model(sentence_in, batch_sent_lens, gpu, debug=debug)
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

            if args.loss_flag == "nlloss":
                loss = loss_function(tag_scores, targets.view(-1)) + loss_function(tag_scores_iden, iden_targets.view(-1))
            elif args.loss_flag == "cross-entropy":
                loss = loss_function(tag_space, targets.view(-1)) + loss_function(tag_space_iden, iden_targets.view(-1))
            loss.backward()
            optimizer.step()
            training_id += sentence_in.size(0)
            if args.batch_size>1 and iteration+1 % 1000 == 0:
                print "## training id in batch", iteration, " is :", training_id, time.asctime()
            if training_id % 50 != 0: continue
            # record best result on dev after each batch training
            loss_dev, prf_dev, prf_dev_iden = eval_model(dev_loader, model, loss_function, "dev", gpu)
            if prf_dev[2] > best_f1:
                print "##-- New best dev results on epoch iter", epoch, iteration, training_id, Tab, best_f1, "(old best)", Tab, loss_dev, time.asctime(), Tab,
                best_f1 = prf_dev[2]
                best_epoch = epoch
                torch.save(model, model_path)
            else:
                print "##-- dev results on epoch iter", epoch, iteration, training_id, Tab, best_f1, "(best f1)", Tab, loss_dev, time.asctime(), Tab,
            outputPRF(prf_dev)
            print "## Iden result:",
            outputPRF(prf_dev_iden)


        ## output result on train
        #loss_train, prf_train, prf_train_iden = eval_model(train_loader, model, loss_function, "train", gpu)
        #print "## train results on epoch:", epoch, Tab, loss_train, time.asctime(), Tab,
        #outputPRF(prf_train)
        #print "## Iden result:", 
        #outputPRF(prf_train_iden)
        ##outputPRF(prf_train_iden[0]), outputPRF(prf_train_iden[1])

# record best result on dev
        loss_dev, prf_dev, prf_dev_iden = eval_model(dev_loader, model, loss_function, "dev", gpu)
        if prf_dev[2] > best_f1:
            print "##-- New best dev results on epoch", epoch, Tab, best_f1, "(old best)", Tab, loss_dev, time.asctime(), Tab,
            best_f1 = prf_dev[2]
            best_epoch = epoch
            torch.save(model, model_path)
        else:
            print "##-- dev results on epoch", epoch, Tab, best_f1, "(best f1)", Tab, loss_dev, time.asctime(), Tab,
        outputPRF(prf_dev)
        print "## Iden result:",
        outputPRF(prf_dev_iden)
        #if best_f1 == 100.0: break
        if best_f1 == 100.0 or (epoch-best_epoch > 50): break
        #outputPRF(prf_dev_iden[0]), outputPRF(prf_dev_iden[1])

# result on test
        if epoch >= 2 and epoch % 10 == 0:
            if epoch % 50 == 0:
                model_test = torch.load(model_path)
                loss_test, prf_test, prf_test_iden = eval_model(test_loader, model_test, loss_function, "test_final", gpu, vocab=vocab, tags_data=tags_data)
                model_test = None
            else:
                loss_test, prf_test, prf_test_iden = eval_model(test_loader, model, loss_function, "test", gpu)
            print "##-- test results on epoch", epoch, Tab, loss_test, time.asctime(), Tab,
            outputPRF(prf_test)
            print "## Iden result:",
            outputPRF(prf_test_iden)
            #outputPRF(prf_test_iden[0]), outputPRF(prf_test_iden[1])

# final result on test
    model = torch.load(model_path)
    loss_train, prf_train, prf_train_iden = eval_model(train_loader, model, loss_function, "test_final", gpu, vocab=vocab, tags_data=tags_data)
    loss_test, prf_test, prf_test_iden = eval_model(test_loader, model, loss_function, "test_final", gpu, vocab=vocab, tags_data=tags_data)
    print "## Final results on test", loss_test, time.asctime(), Tab,
    outputPRF(prf_test)
    print "## Iden result:",
    outputPRF(prf_test_iden)
    #outputPRF(prf_test_iden[0]), outputPRF(prf_test_iden[1])

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
            dev_data = training_data[-500:]
            training_data = training_data[:-500]
            print "first example of dev", dev_data[0]
    model_path = model_path + "_" + time.strftime("%Y%m%d%H%M%S", time.gmtime()) + "_"

    vocab_size = len(vocab)
    pretrain_vocab_size, pretrain_embed_dim = pretrain_embedding.shape
    tagset_size = len(tags_data)

    #sys.exit(0)
    if 0:
        train_sents = [" ".join([str(witem) for witem in item[0]]) for item in training_data]
        dev_sents = [" ".join([str(witem) for witem in item[0]]) for item in dev_data]
        common_in_dev = [item for item in dev_sents if item in train_sents]
        print "## sents in dev, already in train", len(common_in_dev)
        print common_in_dev
        sys.exit(0)
    if 0: # statistic sent length
        all_data = training_data+dev_data+test_data
        sent_lens = [len(item[0]) for item in all_data]
        print "## Statistic sent length:", max(sent_lens), min(sent_lens)
        sys.exit(0)
    if 0: # output to dynet format
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

    #### Batch mode
    if args.batch_size == 1:
        (train_use_tensor, train_use_pad, test_use_tensor, test_use_pad) = (True, False, True, False)
    else:
        (train_use_tensor, train_use_pad, test_use_tensor, test_use_pad) = (False, False, False, False)

    drop_last = False
    train_dataset = MyDataset(training_data, use_tensor=train_use_tensor, use_pad=train_use_pad)
    dev_dataset = MyDataset(dev_data, use_tensor=test_use_tensor, use_pad=test_use_pad)
    test_dataset = MyDataset(test_data, use_tensor=test_use_tensor, use_pad=test_use_pad)
    if train_use_tensor:
        train_loader = torch_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train)
    else:
        train_loader = torch_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle_train, collate_fn=pad_batch, drop_last=drop_last)
    if test_use_tensor:
        dev_loader  = torch_data.DataLoader(dev_dataset, batch_size=args.batch_size)
        test_loader = torch_data.DataLoader(test_dataset, batch_size=args.batch_size)
    else:
        dev_loader  = torch_data.DataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=pad_batch, drop_last=drop_last)
        test_loader = torch_data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=pad_batch, drop_last=drop_last)

    #dev_loader = torch_data.DataLoader(train_dataset, batch_size=args.batch_size)
    #test_loader = None
    data_sets = train_loader, dev_loader, test_loader, pretrain_embedding, vocab, tags_data

    # begin to train
    train_func(para_arr, args, data_sets)

