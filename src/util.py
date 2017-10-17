
from collections import Counter
import numpy as np
import random
import time
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

Tab = "\t"

def outputParameters(para_arr, args):
    vocab_size, tagset_size, embedding_dim = para_arr[:3]
    training_size, dev_size, test_size = para_arr[3:6]
    model_path = para_arr[6]

    print "----- train size         -----", Tab, Tab, training_size
    print "----- dev size           -----", Tab, Tab, dev_size
    print "----- test size          -----", Tab, Tab, test_size
    print "----- vocab size         -----", Tab, Tab, vocab_size
    print "----- tags size          -----", Tab, Tab, tagset_size
    print "----- batch size         -----", Tab, Tab, args.batch_size

    print "----- bilstm             -----", Tab, Tab, args.bilstm
    print "----- embeds dim         -----", Tab, Tab, embedding_dim
    print "----- hidden dim         -----", Tab, Tab, args.hidden_dim
    print "----- hidd dim 2nd       -----", Tab, Tab, args.hidden_dim_snd
    print "----- layers num         -----", Tab, Tab, args.num_layers
    print "----- dropout            -----", Tab, Tab, args.dropout
    print "----- learn rate         -----", Tab, Tab, args.lr
    print "----- #iteration         -----", Tab, Tab, args.epoch_num
    print "----- use gpu            -----", Tab, Tab, args.gpu
    print "----- test as dev        -----", Tab, Tab, args.test_as_dev
    print "----- shuf train         -----", Tab, Tab, args.shuffle_train
    print "----- use conv           -----", Tab, Tab, args.use_conv
    print "----- use position       -----", Tab, Tab, args.use_position
    if not args.use_conv: return
    print "----- conv1 width        -----", Tab, Tab, args.conv_width1
    print "----- conv2 width        -----", Tab, Tab, args.conv_width2
    print "----- #conv filter       -----", Tab, Tab, args.conv_filter_num


def outputPRF(arr):
    arr = ["%.2f"%i for i in arr]
    print "-- Pre, Rec, F1:", Tab, arr[0], Tab, arr[1], Tab, arr[2]

def loadVocab(filename):
    content = open(filename, "r").readlines()
    words = [w.strip() for w in content]
    vocab = dict(zip(words, range(len(words))))
    vocab["unk"] = len(vocab)   # not sure about 19488, not appeared in training
    vocab["<unk>"] = len(vocab) # 19489 is unk
    return vocab

def loadTag(filename):
    content = open(filename, "r").readlines()
    words = [w.strip() for w in content]
    vocab = dict(zip(words, range(1, len(words)+1)))
    vocab["NULL"] = 0
    return vocab

def loadPretrain2(model_path):
    content = open(model_path, "r").readlines()
    wordNum, dim = content[0].strip().split()
    wordNum = int(wordNum)
    dim = int(dim)

    pretrain_embedding = []
    pretrain_vocab = {} # word: word_id
    for word_id, line in enumerate(content[1:]):
        word_item = line.strip().split()
        word_text = word_item[0]
        embed_word = [float(item) for item in word_item[1:]]
        pretrain_embedding.append(embed_word)
        pretrain_vocab[word_text] = word_id
    return np.matrix(pretrain_embedding), pretrain_vocab

def loadPretrain(model_path):
    content = open(model_path, "r").readlines()
    pretrain_embedding = []
    for line in content:
        embed_word = line.strip().split(",")
        embed_word = [float(item) for item in embed_word]
        pretrain_embedding.append(embed_word)
    pretrain_embed_dim = len(pretrain_embedding[0])
    pretrain_embedding.append([random.uniform(-1, 1) for _ in range(pretrain_embed_dim)])
    pretrain_embedding.append([random.uniform(-1, 1) for _ in range(pretrain_embed_dim)])
    return np.matrix(pretrain_embedding)

# load train dev test
def loadTrainData2(filename):
    content = open(filename, "r").readlines()
    content = [line.strip().lower() for line in content if len(line.strip())>1]
    data = [(line.split("\t")[0].strip().split(), line.split("\t")[1].strip().split()) for line in content]

    data = [item for item in data if len(item[0]) > 3 and len(item[0])<80]
    if len(data) != len(content): print "-- length new ori", len(data), len(content)
    for sent_id, sent_item in enumerate(data):
        if len(sent_item[0]) < 1 or len(sent_item[1]) < 1 or len(sent_item[0]) != len(sent_item[1]):
            print "## Error!! loading data:", sent_id, len(sent_item[0]), len(sent_item[1]), content[sent_id]
            print sent_item
            break
    return data

# load train dev test
def loadTrainData(filename):
    content = open(filename, "r").readlines()
    content = [line.strip() for line in content if len(line.strip())>1]
    data = [(line.split("\t")[0].strip().split(), line.split("\t")[1].strip().split()) for line in content]
    data = [([int(item) for item in sent], [int(item) for item in tag]) for sent, tag in data if len(sent) < 80]
    #data = [(sent, [0 if item == 0 else 1 for item in tag]) for sent, tag in data]
    if len(data) != len(content): print "-- length new ori", len(data), len(content)
    for sent_id, sent_item in enumerate(data):
        if len(sent_item[0]) < 1 or len(sent_item[1]) < 1 or len(sent_item[0]) != len(sent_item[1]):
            print content[sent_id]
            print sent_item
    return data

def output_dynet_format(data, vocab, tags_data, filename):
    id2tag = dict([(tag_index, tag) for tag, tag_index in tags_data.items()])
    id2tag[0] = "O"
    sep = "\t"
    id2word = dict([(word_index, word) for word, word_index in vocab.items()])
    outFile = file(filename, "w")
    tag_appear = []
    for sent, tags in data:
        tag_appear.extend(tags)
        words = [id2word.get(word_index) for word_index in sent]
        for word, tag in zip(words, tags):
            #output_str = word + sep + "NN" + sep + "NN" + sep + str(tag) + sep + "1"
            tag_text = id2tag[tag]
            output_str = word + sep + tag_text
            outFile.write(output_str + "\n")
        outFile.write("\n")
    tag_counter = Counter(tag_appear)
    tag_appear = sorted(tag_counter.items(), key = lambda a:a[0])
    print tag_appear
    outFile.close()

def output_normal_pretrain(pretrain_embedding, vocab, filename):
    sep = " "
    print "--vocab_size:", len(vocab), " dim:", pretrain_embedding.shape
    outFile = file(filename, "w")
    id2word = dict([(word_index, word) for word, word_index in vocab.items()])
    vocab_size, dim = pretrain_embedding.shape
    outFile.write(str(vocab_size)+sep+str(dim) + "\n")
    for word_index, embed in enumerate(pretrain_embedding.tolist()):
        word = id2word[word_index]
        embed_str = sep.join(["%.6f"%val for val in embed])
        output_str = word + sep + embed_str + sep
        outFile.write(output_str + "\n")
    outFile.close()

def check_trigger_test(training_data, test_data):
    test_triggers_index = [word_index for sent, tags in test_data for word_index, tag in zip(sent, tags) if tag != 0]
    test_words_index = [word_index for sent, tags in test_data for word_index, tag in zip(sent, tags)]
    train_triggers_index = [word_index for sent, tags in training_data for word_index, tag in zip(sent, tags) if tag != 0]
    train_words_index = [word_index for sent, tags in training_data for word_index, tag in zip(sent, tags)]
    test_only_triggers = list(set(test_triggers_index)-set(train_triggers_index))
    print len(test_only_triggers), sorted(test_only_triggers)
    test_only_words = list(set(test_triggers_index)-set(train_words_index))
    print len(test_only_words), sorted(test_only_words)
    test_only_words = list(set(test_words_index)-set(train_words_index))
    print len(test_only_words), sorted(test_only_words)

def check_trigger(sent_tags, tagnum=33):
    sent_tags_str = " ".join([str(i) for i in sent_tags])
    #multi_tags = [str(i)+" "+str(i)+" "+str(i) for i in range(1, tagnum+1)]
    #multi_tags = [str(i)+" "+str(i) for i in range(1, tagnum+1)]
    multi_tags = [str(i)+" "+str(j) for i in range(1, tagnum+1) for j in range(1, tagnum+1)]
    for multi_tag_str in multi_tags:
        if sent_tags_str.find(multi_tag_str) >= 0:
            print "-- multi-word trigger", sent_tags

def check_data(data, vocab):
    id2word = dict([(word_index, word) for word, word_index in vocab.items()])
# df = 1 > unk
    vocab_index = [word_index for item in data for word_index in item[0]]
    counter_index = Counter(vocab_index)
    vocab_size = len(counter_index)
    word_df_one = [word_index for word_index, df in counter_index.items() if df == 1]
    #print min(vocab_index), counter_index[min(vocab_index)], max(vocab_index), counter_index[max(vocab_index)], min(counter_index.values()), max(counter_index.values())
    new_data = [([word_index if word_index not in word_df_one else vocab_size-1 for word_index in sent], tags) for sent, tags in data]
    for sent, tags in data:
        if len(sent) < 1 or len(tags) < 1:
            print "-- 0-length data", sent, tags
        if sum(tags) == 0: continue
        sent_text = [id2word.get(word_index) for word_index in sent]
        #print sent
        #print sent_text
        #print tags
        print zip(sent, sent_text, tags)
        print "--trigger:",
        for word_index, word, tag in zip(sent, sent_text, tags):
            if word_index == 19489 and tag != 0:
                print "-- unk trigger"
                break
            if tag != 0:
                print (word_index, word, tag), Tab,
        print
    return new_data


def get_trigger(sent_tags):
    triggers = [(word_idx, tag) for word_idx, tag in enumerate(sent_tags) if tag != 0]
    #triggers_multiword = [((trigger_idx, trigger_idx+1), tag) for trigger_idx, trigger in enumerate(triggers[:-1]) if triggers[trigger_idx+1][1] == trigger[1]]
    #return triggers_multiword
    return triggers

def evalPRF_iden(items_in_docs_gold, items_in_docs):
    debug = False
    common_in_docs = []
    num_in_docs_gold = []
    num_in_docs = []
    for items_in_doc, items_in_doc_gold in zip(items_in_docs_gold, items_in_docs):
        match_result_in_doc = [eval_sysResult(item, items_in_doc_gold, "iden") for item in items_in_doc]
        common_in_doc = [item for item, match_result in zip(items_in_doc, match_result_in_doc) if match_result is not None]

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

    pre, rec, f1 = calPRF(common, num, num_gold)
    return pre, rec, f1

def eval_sysResult(sys_out, gold_outs, eval_flag="class"):
    if eval_flag == "iden":
        matched = [gold_out for gold_out in gold_outs if sys_out[0] == gold_out[0]]
    else:
        matched = [gold_out for gold_out in gold_outs if sys_out == gold_out]
    if len(matched) == 0: return None
    return matched[0]

def evalPRF(items_in_docs_gold, items_in_docs, data_flag="train"):
    debug = False
    common_in_docs = []
    num_in_docs_gold = []
    num_in_docs = []
    for items_in_doc, items_in_doc_gold in zip(items_in_docs_gold, items_in_docs):
        match_result_in_doc = [eval_sysResult(item, items_in_doc_gold) for item in items_in_doc]
        common_in_doc = [item for item, match_result in zip(items_in_doc, match_result_in_doc) if match_result is not None]
        missed_in_doc_gold = [item for item in items_in_doc_gold if item not in common_in_doc]
        wrong_in_doc = [item for item in items_in_doc if item not in common_in_doc]
        if data_flag == "test_final":
            print "## final results of test doc:", len(common_in_docs)+1, " common, wrong, missed", common_in_doc, wrong_in_doc, missed_in_doc_gold

        common_in_docs.append(len(common_in_doc))
        num_in_docs_gold.append(len(items_in_doc_gold))
        num_in_docs.append(len(items_in_doc))

    common = sum(common_in_docs)
    num_gold = sum(num_in_docs_gold)
    num = sum(num_in_docs)

    if data_flag == "test_final": debug = True
    if debug:
        print "-- common, num_gold, num:", common, num_gold, num
        #print "-- common_in_docs", common_in_docs
        #print "-- num_in_docs_gold", num_in_docs_gold
        #print "-- num_in_docs", num_in_docs

    pre, rec, f1 = calPRF(common, num, num_gold)
    return pre, rec, f1

def calPRF(common, num, num_gold):
    if common == 0: return 0.0, 0.0, 0.0
    pre = common*100.0/num
    rec = common*100.0/num_gold
    f1 = 2*pre*rec/(pre+rec)
    return pre, rec, f1


def load_data2(args):
    debug = True

    pretrain_embedding, pretrain_vocab = loadPretrain2(args.pretrain_embed)
    print "## pretrained embedding loaded.", time.asctime(), pretrain_embedding.shape

    training_data = loadTrainData2(args.train)
    print "## train loaded.", args.train, time.asctime()
    dev_data = loadTrainData2(args.dev)
    print "## dev loaded.", args.dev, time.asctime()
    test_data = loadTrainData2(args.test)
    print "## test loaded.", args.test, time.asctime()
    all_data = training_data + dev_data + test_data
    vocab = sorted(list(set([word_text for sent_text, _ in all_data for word_text in sent_text])))
    vocab = dict(zip(vocab, range(len(vocab))))
    vocab["<unk>"] = len(vocab)
    #pretrain_vocab = vocab
    #pretrain_embedding = np.random.uniform(-1.0, 1.0, (len(pretrain_vocab), 300))
    unk_id = len(pretrain_vocab)-1

    id2word = dict([(pretrain_vocab[item], item) for item in pretrain_vocab])
    if debug:
        for i in range(10, 13):
            sent_tags = training_data[i][1]
            triggers = [(word_idx, training_data[i][0][word_idx], tag) for word_idx, tag in enumerate(sent_tags) if tag != "none"]
            print "## eg:", training_data[i]
            print triggers
    tags_data = sorted(list(set([tag_text for sent_text, tags_text in training_data for tag_text in tags_text if tag_text != "none"])))
    tags_data = dict(zip(tags_data, range(1, len(tags_data)+1)))
    #tags_data = dict([(tag_text, tag_id+1) for tag_id, tag_text in enumerate(tags_data)])
    tags_data["none"] = 0

    ## text to id
    training_data = [([pretrain_vocab[word_text] if word_text in pretrain_vocab else unk_id for word_text in sent_text_arr], [tags_data.get(tag_text) for tag_text in tags_text_arr]) for sent_text_arr, tags_text_arr in training_data]
    dev_data = [([pretrain_vocab[word_text] if word_text in pretrain_vocab else unk_id for word_text in sent_text_arr], [tags_data.get(tag_text) for tag_text in tags_text_arr]) for sent_text_arr, tags_text_arr in dev_data]
    test_data = [([pretrain_vocab[word_text] if word_text in pretrain_vocab else unk_id for word_text in sent_text_arr], [tags_data.get(tag_text) for tag_text in tags_text_arr]) for sent_text_arr, tags_text_arr in test_data]
    if debug:
        for i in range(10, 13):
            sent_tags = training_data[i][1]
            triggers = [(word_idx, id2word[training_data[i][0][word_idx]], tag) for word_idx, tag in enumerate(sent_tags) if tag != 0]
            print "## eg:", training_data[i]
            print triggers
    return training_data, dev_data, test_data, pretrain_vocab, tags_data, pretrain_embedding, args.model

# resize train dev test
def resizeVocab(train_data, test_data, vocab, pretrain_embedding):
    old_vocab_size = len(vocab)
    word_counter = Counter()
    for sent, _ in train_data + test_data:
        word_counter += Counter(sent)
    words_top = [word for word, word_num in word_counter.most_common() if word_num >= 2]
    if old_vocab_size-1 not in words_top: words_top.append(old_vocab_size-1)
    new_vocab_size = len(words_top)

    words_change = {}
    for i in range(old_vocab_size):
        if i in words_top:
            words_change[i] = len(words_change)
    train_data = [([words_change[word] if word in words_top else new_vocab_size-1 for word in sent], tag) for sent, tag in train_data]
    test_data = [([words_change[word] if word in words_top else new_vocab_size-1 for word in sent], tag) for sent, tag in test_data]
    #print len(words_top)
    words_del = [i for i in range(old_vocab_size) if i not in words_top]
    pretrain_embedding = np.delete(pretrain_embedding, words_del, 0)
    vocab = dict([(wstr, words_change[wid]) for wstr, wid in vocab.items() if wid in words_top])
    return train_data, test_data, vocab, pretrain_embedding

def sort_data(dataset):
    sent_length = [(sent_id, len(item[0])) for sent_id, item in enumerate(dataset)]
    sorted_sent_length = sorted(sent_length, key = lambda a:a[1], reverse=True)
    new_dataset = []
    for item in sorted_sent_length:
        new_dataset.append(dataset[item[0]])
    return new_dataset

def load_data(args):
# pretrain embedding: matrix (vocab_size, pretrain_embed_dim)
    pretrain_embedding = loadPretrain(args.pretrain_embed)
    print "## pretrained embedding loaded.", time.asctime(), pretrain_embedding.shape

# vocab: word: word_id
    vocab = loadVocab(args.vocab)
    print "## vocab loaded.", time.asctime()

# train test
    training_data = loadTrainData(args.train)
    print "## train loaded.", args.train, time.asctime()
    #training_data = check_data(training_data, vocab)
    test_data = loadTrainData(args.test)
    print "## test loaded.", args.test, time.asctime()
    #test_data = check_data(test_data, vocab)
    #check_trigger_test(training_data, test_data)

    training_data, test_data, vocab, pretrain_embedding = resizeVocab(training_data, test_data, vocab, pretrain_embedding)

# tags_data: tag_name: tag_id
    tags_data = loadTag(args.tag)
    print "## event tags loaded.", time.asctime()

    #for sent, tag in training_data:
    #    check_trigger(tag)
    #for sent, tag in test_data:
    #    check_trigger(tag)
    return training_data, None, test_data, vocab, tags_data, pretrain_embedding, args.model

# test of dataloader
def check_dataloader(dataloader):
    for iteration, batch in enumerate(dataloader):
        sentence_in, targets, batch_sent_lens = batch # tensors, tensors, arr
        batch_sent_lens, sorted_lens_idx = batch_sent_lens.sort(dim=0, descending=True)
        sentence_in = sentence_in[sorted_lens_idx]
        print batch_sent_lens.size(), len(batch_sent_lens.numpy())
        print sentence_in.size()
        print sentence_in
        print batch_sent_lens.numpy()
        sentence_in_pack = pack_padded_sequence(sentence_in, batch_sent_lens.numpy(), batch_first=True)

        continue
        for target_doc in targets:
            print "eval target doc", target_doc.numpy().tolist()
            gold_triggers = get_trigger(target_doc.numpy().tolist())
            print gold_triggers

