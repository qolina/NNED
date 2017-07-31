
Tab = "\t"

def outputParameters(para_arr):
    vocab_size, tagset_size, embedding_dim, hidden_dim = para_arr[:4]
    dropout, bilstm, num_layers, gpu, iteration_num, learning_rate = para_arr[4:10]
    training_size, dev_size, test_size = para_arr[10:13]
    print "----- train size -----", Tab, Tab, training_size
    print "----- dev size   -----", Tab, Tab, dev_size
    print "----- test size  -----", Tab, Tab, test_size
    print "----- vocab size -----", Tab, Tab, vocab_size
    print "----- tags size  -----", Tab, Tab, tagset_size
    print "----- embeds dim -----", Tab, Tab, embedding_dim
    print "----- hidden dim -----", Tab, Tab, hidden_dim
    print "----- layers num -----", Tab, Tab, num_layers
    print "----- #iteration -----", Tab, Tab, iteration_num
    print "----- dropout    -----", Tab, Tab, dropout
    print "----- learn rate -----", Tab, Tab, learning_rate
    print "----- bilstm     -----", Tab, Tab, bilstm
    print "----- use gpu    -----", Tab, Tab, gpu


def outputPRF(arr):
    arr = ["%.2f"%i for i in arr]
    print "-- Pre, Rec, F1:", Tab, arr[0], Tab, arr[1], Tab, arr[2]

def loadVocab(filename):
    content = open(filename, "r").readlines()
    words = [w.strip() for w in content]
    vocab = dict(zip(words, range(1, len(words)+1)))
    return vocab

def loadPretrain(model_path):
    content = open(model_path, "r").readlines()
    pretrain_embedding = []
    for line in content:
        embed_word = line.strip().split(",")
        embed_word = [float(item) for item in embed_word]
        pretrain_embedding.append(embed_word)
    pretrain_embed_dim = len(pretrain_embedding[0])
    pretrain_embedding.append([random.uniform(-1, 1) for _ in range(pretrain_embed_dim)])
    return np.matrix(pretrain_embedding)

# load train dev test
def loadTrainData(filename):
    content = open(filename, "r").readlines()
    content = [line.strip() for line in content if len(line.strip())>1]
    data = [(line.split("\t")[0].strip().split(), line.split("\t")[1].strip().split()) for line in content]
    data = [([int(item)-1 for item in sent], [int(item) for item in tag]) for sent, tag in data]
    if len(data) != len(content): print "-- length new ori", len(data), len(content)
    for sent_id, sent_item in enumerate(data):
        if len(sent_item[0]) < 1 or len(sent_item[1]) < 1 or len(sent_item[0]) != len(sent_item[1]):
            print content[sent_id]
            print sent_item
    return data


