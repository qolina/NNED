#!/bin/sh

# feng
#CUDA_VISIBLE_DEVICES=0 python trigger_lstm.py -train ../ni_data/pre_processed_feng/tmp.train -test ../ni_data/pre_processed_feng/tmp.test  -tag ../ni_data/pre_processed_feng/labellist -pretrain_embed ../ni_data/pre_processed_feng/wordvector -vocab ../ni_data/pre_processed_feng/wordlist -model ../ni_data/models/model.feng.trigger --batch_size 10 --no_use_pos --no_use_conv --test_as_dev --lr 0.5 --dropout 0.1 
# --opti_flag adadelta

#CUDA_VISIBLE_DEVICES=3 python trigger_lstm.py -train ../ni_data/pre_processed_feng/tmp.train -test ../ni_data/pre_processed_feng/tmp.test  -tag ../ni_data/pre_processed_feng/labellist -pretrain_embed ../ni_data/pre_processed_feng/wordvector -vocab ../ni_data/pre_processed_feng/wordlist -model ../ni_data/models/model.trigger

# qin corpus
#CUDA_VISIBLE_DEVICES=0 python trigger_lstm.py -train ../ni_data/EngMix/train.triggerEvent.txt -dev ../ni_data/EngMix/dev.triggerEvent.txt -test ../ni_data/EngMix/test.triggerEvent.txt -pretrain_embed ../ni_data/ace.pretrain300.vectors -model ../ni_data/models/model.qin.trigger --no_use_pos --no_use_conv --batch_size 10 --lr 0.5 --dropout 0.1

#CUDA_VISIBLE_DEVICES=3 python trigger_lstm.py -train ../ni_data/EngMix/train.triggerEvent.txt -dev ../ni_data/EngMix/dev.triggerEvent.txt -test ../ni_data/EngMix/test.triggerEvent.txt -pretrain_embed ../ni_data/sskip.100.vectors -model ../ni_data/models/model.trigger

# liqi corpus
CUDA_VISIBLE_DEVICES=0 python trigger_lstm.py -train ../ni_data/liqi_zhou_data/train.trig.lq.txt -dev ../ni_data/liqi_zhou_data/dev.trig.lq.txt -test ../ni_data/liqi_zhou_data/test.trig.lq.txt -pretrain_embed ../ni_data/ace.pretrain300.vectors -model ../ni_data/models/model.liqi.trigger --no_use_pos --no_use_conv --batch_size 10 --lr 0.5 --dropout 0.1
