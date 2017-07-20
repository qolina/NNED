# This file mainly implements a basic attention model for neural event extraction
# input file consists of sentences which contain one-event.
import os
import re
import sys
import random
from collections import Counter
from aceEventUtil import str2ArgTrain

from nltk.tokenize import sent_tokenize, word_tokenize

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

class LSTMAttentionArg(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMAttentionArg, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
        return 

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def loadWord2Vec(modelPath):
   content = open(modelPath, "r").readlines()
   wordNum, dim = content[0].strip().split()
   content = [line.strip().split() for line in content[1:]]

   words = [item[0] for item in content]
   embeddings = [[float(val) for val in item[1:]] for item in content]

   #wordIDHash = {word:wordID}
   wordIDHash = dict(zip(words, range(wordNum)))

   # add unk
   unk_word, unk_vector = ("<unk>", [random.random() for _ in range(dim)])
   wordIDHash[unk_word] = wordNum
   embeddings.append(unk_vector)

   return len(wordIDHash), dim, wordIDHash, embeddings

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
    return [arg1, arg2]


if __name__ == "__main__":
    print "Usage: python statisticCorpus.py -train trainFile -embed embeddingFile"
    print sys.argv
    trainFilename, embeddingFilename = parseArgs(sys.argv)

    vocab_size, Embedding_dim, Vocab, EmbeddingArr = loadWord2Vec(embeddingFilename)
    content = open(trainFilename, "r").readlines()
    eventArr = [trainFormatStr2event_noType(line.strip(), "|||", "\t") for line in content]
    training_data = prepTrain(eventArr)

    Hidden_dim = 100
    model = LSTMAttentionArg(Embedding_dim, Hidden_dim, vocab_size)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(100):
        for sentence, _ in training_data:
            model.zero_grad()
            model.hidden = model.init_hidden()

            sentence_in = prepare_sequence(sentence, wordIDHash)

            loss = loss_function()
            loss.backward()
            optimizer.step()
