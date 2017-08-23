#!/bin/sh

# feng
CUDA_VISIBLE_DEVICES=3 python trigger_lstm.py -train ../ni_data/pre_processed_feng/tmp.train -test ../ni_data/pre_processed_feng/tmp.test  -tag ../ni_data/pre_processed_feng/labellist -embed ../ni_data/pre_processed_feng/wordvector -vocab ../ni_data/pre_processed_feng/wordlist -model ../ni_data/models/model.trigger

# qin corpus
#CUDA_VISIBLE_DEVICES=3 python trigger_lstm.py -train ../ni_data/EngMix/train.triggerEvent.txt -dev ../ni_data/EngMix/dev.triggerEvent.txt -test ../ni_data/EngMix/test.triggerEvent.txt -embed ../ni_data/ace.pretrain300.vectors -model ../ni_data/models/model.trigger

#CUDA_VISIBLE_DEVICES=3 python trigger_lstm.py -train ../ni_data/EngMix/train.triggerEvent.txt -dev ../ni_data/EngMix/dev.triggerEvent.txt -test ../ni_data/EngMix/test.triggerEvent.txt -embed ../ni_data/sskip.100.vectors -model ../ni_data/models/model.trigger
