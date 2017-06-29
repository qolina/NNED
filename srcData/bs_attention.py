# This file mainly implements a basic attention model for neural event extraction
# input file consists of sentences which contain one-event.
import os
import re
import sys
from collections import Counter
from readEventTag import event2str
from statisticCorpus import str2event

import torch
from torch.autograd import Variable

def loadWord2Vec(modelPath):
   content = open(modelPath, "r").readlines()
   wordNum, dim = content[0].strip().split()
   content = [line.strip().split() for line in content[1:]]
   content = [(item[0], [float(val) for val in item[1:]]) for item in content]
   #word2vecModel = {} #word:vector
   word2vecModel = dict(content)
   return word2vecModel

def prepData(eventArr, Vocab):
    for event in eventArr:

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


# event = [eventtype, eventsubtype, sentence_ldc_scope, anchor, (arg, role), (arg, role), ...]
# eventstring: sentence[sep]eventtype[sep]eventsubtype[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
if __name__ == "__main__":
    print "Usage: python statisticCorpus.py -train trainFile -embed embeddingFile"
    print sys.argv

    Vocab = loadWord2Vec(sys.argv[2])
    content = open(sys.argv[1], "r").readlines()
    eventArr = [str2event(line.strip(), "\t") for line in content]
