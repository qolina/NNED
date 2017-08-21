
from collections import Counter
import numpy as np
import random

Tab = "\t"

def outputParameters(para_arr):
    vocab_size, tagset_size, embedding_dim, hidden_dim = para_arr[:4]
    dropout, bilstm, num_layers, gpu, iteration_num, learning_rate = para_arr[4:10]
    training_size, dev_size, test_size = para_arr[10:13]
    conv_width1, conv_width2, conv_filter_num, hidden_dim_snd = para_arr[13:17]
    print "----- train size -----", Tab, Tab, training_size
    print "----- dev size   -----", Tab, Tab, dev_size
    print "----- test size  -----", Tab, Tab, test_size
    print "----- vocab size -----", Tab, Tab, vocab_size
    print "----- tags size  -----", Tab, Tab, tagset_size
    print "----- embeds dim -----", Tab, Tab, embedding_dim
    print "----- hidden dim -----", Tab, Tab, hidden_dim
    print "----- hidd dim 2nd-----", Tab, Tab, hidden_dim_snd
    print "----- conv1 width -----", Tab, Tab, conv_width1
    print "----- conv2 width -----", Tab, Tab, conv_width2
    print "----- #conv filter -----", Tab, Tab, conv_filter_num
    print "----- layers num -----", Tab, Tab, num_layers
    print "----- dropout    -----", Tab, Tab, dropout
    print "----- learn rate -----", Tab, Tab, learning_rate
    print "----- #iteration -----", Tab, Tab, iteration_num
    print "----- bilstm     -----", Tab, Tab, bilstm
    print "----- use gpu    -----", Tab, Tab, gpu


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

    data = [item for item in data if len(item[0]) > 3]
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
    data = [([int(item) for item in sent], [int(item) for item in tag]) for sent, tag in data]
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
        common_in_doc = [1 for item_gold, item in zip(items_in_doc_gold, items_in_doc) if item_gold[0] == item[0]]

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

def evalPRF(items_in_docs_gold, items_in_docs):
    debug = False
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

    pre, rec, f1 = calPRF(common, num, num_gold)
    return pre, rec, f1

def calPRF(common, num, num_gold):
    if common == 0: return 0.0, 0.0, 0.0
    pre = common*100.0/num
    rec = common*100.0/num_gold
    f1 = 2*pre*rec/(pre+rec)
    return pre, rec, f1



