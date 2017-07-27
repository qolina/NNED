import os
import re
import sys
import random
import cPickle
from collections import Counter
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

from aceEventUtil import event2str, str2event, event2TrainFormatStr_noType, loadEventHierarchy, argTrain2str, outputEventFormat2

Tab = "\t"

# event_v2 = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
def indexEventBySentence(eventArr, allSents, negSentNum):
    sentHash = {}
    for event in eventArr:
        sentence, _ = event[0]
        if sentence in sentHash:
            sentHash[sentence].append(event)
        else:
            sentHash[sentence] = [event]

    negSents = [line for line in allSents if line not in sentHash]
    oneEventSentence = dict([(sent, events[0]) for sent, events in sentHash.items() if len(events) == 1])
    eventNumInSent = [len(events) for sent, events in sentHash.items()]
    sentNumAll = len(sentHash)
    print "## ori #event", len(eventArr)
    print "## #sentence-with-events", sentNumAll
    print "## #sentences", len(allSents), "#negSents", len(negSents)
    print "## #events-in-sents:",
    print Counter(eventNumInSent).most_common()
    for eventNum, sentNum in Counter(eventNumInSent).most_common():
        print eventNum, "\t", sentNum, "\t", sentNum*100.0/sentNumAll

    return oneEventSentence, random.sample(negSents, min(negSentNum, len(negSents)))

def extractSpan_word(words, wordsArg):
    strWhole = " ".join(words)
    strPart = " ".join(wordsArg)
    index = strWhole.find(strPart)
    preWordNum = len(strWhole[:index].strip().split())
    return (preWordNum, preWordNum+len(wordsArg)-1)

def extractSpan_str(sentence, arg):
    sentText, sentIndex = sentence
    argText, argIndex = arg
    (st, ed) = (1, 1)
    if argIndex[0]>0 and sentText[argIndex[0]-1] != " ":
        print "##Wrong arg charSpan", arg, sentence
    if argIndex[1]<sentIndex[1]-1 and sentText[argIndex[1]+1] != " ":
        print "##Wrong arg charSpan", arg, sentence
    return (st, ed)

def negSent2argTrain(negSents, posSentNum):
    role = "trigger"
    eventSubType = "none"
    neg_training_data = []
    for sentId, sent in enumerate(negSents):
        item = ()
        neg_training_data.append((str(sentId + posSentNum), sent, eventSubType, (-1, -1), role))
    return neg_training_data

# eventArr(in format 2) convert to argument training data format
# event (in Format2): [sentence, eventSubType, positive arg-role pair list, negtive role list]
# argumentTrain: (sentenceId, sentence, argumentIndexArr, role)
def events2argTrain(eventArr, testFlag):
    debug = True
    training_data = []
    sentenceArr = []
    for event in eventArr:
        sentence, eventSubType, arg_role_Pos, role_Neg = event
        sentenceArr.append(sentence[0])
        if debug:
            outputEventFormat2(event)
        wordsIn = wordpunct_tokenize(sentence[0])
        sentenceText = " ".join(wordsIn)
        if debug:
            print "-- wordsInSent:", wordsIn
        for arg, role, arg_index in arg_role_Pos:
            wordsInArg = wordpunct_tokenize(arg)
            argSpan = extractSpan_word(wordsIn, wordsInArg)
            #argSpan = extractSpan_str(sentence, (arg, arg_index))
            training_data.append((str(len(sentenceArr)-1), sentenceText, eventSubType, argSpan, role))
            if debug:
                print "-- args, role:", arg, role, arg_index, argSpan
        if testFlag: continue
        for role in role_Neg:
            training_data.append((str(len(sentenceArr)-1), sentenceText, eventSubType, (-1, -1), role))
    return training_data, sentenceArr

def statisticArgData(training_data, Arguments):
    roles = [role.lower() for _, _, _, idxs, role in training_data]
    role_counter = Counter(roles)
    role_lens = [(role.lower(), idxs[1]-idxs[0]+1) for _, _, _, idxs, role in training_data]
    print "## statistic arguments:"
    print role_counter.most_common()
    print "-- role, #instance, #avglen"
    Arguments.add("trigger")
    for role in sorted(list(Arguments)):
        lens = [roleLen for roleItem, roleLen in role_lens if roleItem==role]
        avg_len = np.mean(lens) if len(lens)>0 else 0
        print role_counter[role], Tab, role_counter[role]*100.0/len(training_data), Tab, avg_len, Tab, Tab, role

# event_v1 = [sentence_ldc_scope, eventType, eventSubType, anchorText, (argText, role), (argText, role), ...]
# eventFormat2_v1: [sentence, eventSubType, positive arg-role pair list, negtive role list]
# event_v2 = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
# eventFormat2_v2: [sentence, eventSubType, positive arg-role pair list, negtive role list]   --detail: [(sentence_ldc_scope, index), eventType, eventSubType, [(anchorText, "trigger", index), (argText, role, index), ...], [role, role, ...]]
def event2Format2(event, Arguments, eventSubTypeRoleHash):
    arg_roles_Pos = [(event[3][0], "trigger", event[3][1])]
    arg_roles_Pos.extend(event[4:])
    roles_Pos = [arg[1].lower() for arg in event[4:]]
    #roles_Neg = random.sample(list(Arguments-set(roles_Pos)), 3)
    roles_type = eventSubTypeRoleHash[event[2].lower()]
    roles_Neg = list(set(roles_type)-set(roles_Pos))
    return (event[0], event[2].lower(), arg_roles_Pos, roles_Neg)


def loadArguments(filename):
    eventSubTypeHash, eventSubTypeRoleHash = loadEventHierarchy(filename)
    return set([role for argRoles in eventSubTypeRoleHash.values() for role in argRoles])

if __name__ == "__main__":
    print "Usage: python .py dataDir argumentsFile outputFilename (.../train.oneEventSent.txt)"
    print sys.argv

    testFlag = False
    if sys.argv[3].find("test.oneEventSent") >= 0: testFlag=True
    devFlag = False
    if sys.argv[3].find("dev.oneEventSent") >= 0: devFlag=True
    eventSubTypeHash, eventSubTypeRoleHash = loadEventHierarchy(sys.argv[2])
    print eventSubTypeRoleHash.items()
    #Arguments = loadArguments(sys.argv[2])
    Arguments = set([role for argRoles in eventSubTypeRoleHash.values() for role in argRoles])
    print "## #Arguments", len(Arguments), Arguments

    dataDir = sys.argv[1]
    fileList = sorted(os.listdir(dataDir))
    allEvents = []
    allSents = []
    for filename in fileList:
        if filename.endswith(".sgm"):
            sgmContent = open(dataDir + filename, "r").readlines()
            sgmTextContent = "".join(sgmContent[9:-3]).split("\n\n")
            sents = [line.replace("\n", " ").strip().rstrip(".") for line in sgmTextContent if len(line) > 1]
            allSents.extend(sents)

        if not (os.path.isfile(dataDir + filename) and filename.endswith(".ee")): continue
        #print "## Processing ", filename
        # read event from str
        #content = open(dataDir+filename, "r").readlines()
        #eventArrOneDoc = [str2event(line.strip(), "||||") for line in content]

        # from cPickle
        infile = file(dataDir+filename, "r")
        eventArrOneDoc = cPickle.load(infile)

        allEvents.extend(eventArrOneDoc)

    allSents = list(set(allSents))
    negSentNum = 2000
    if devFlag: negSentNum = len(allSents)
    if testFlag: negSentNum = len(allSents)
    oneEventSentence, negSents = indexEventBySentence(allEvents, allSents, negSentNum)
    eventArr_format2 = [event2Format2(event, Arguments, eventSubTypeRoleHash) for sent, event in oneEventSentence.items()]
    training_data, sentenceArr = events2argTrain(eventArr_format2, testFlag)
    statisticArgData(training_data, Arguments)

    neg_training_data = negSent2argTrain(negSents, len(sentenceArr))

    if not testFlag:
        training_data.extend(neg_training_data)
    outfile = open(sys.argv[3], "w")
    for argTrain in training_data:
        argTrain_str = argTrain2str(argTrain, " ||| ", " ")
        outfile.write(argTrain_str + "\n")

    # written format 1
    #for sent, event in oneEventSentence.items():
    #    eventString = event2TrainFormatStr_noType(event, "|||", "\t", Arguments)
    #    outfile.write(eventString + "\n")
    outfile.close()
    print "## events writen to", outfile.name
