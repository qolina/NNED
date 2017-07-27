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
    debug = True
    sentHash = {}
    for event in eventArr:
        sentence, _ = event[0]
        if sentence in sentHash:
            sentHash[sentence].append(event)
        else:
            sentHash[sentence] = [event]

    if debug:
        print "-- Examples of sents-with-event"
        print "\n".join(sorted(sentHash.keys())[:10])
        print "-- Examples of sents all"
        print "\n".join(sorted(allSents)[:10])
    containEventSents = []
    for line in sentHash:
        appIndex = [li for li, line2 in enumerate(allSents) if line2.find(line) >= 0]
        containEventSents.extend(appIndex)
    containEventSents = Counter(containEventSents)
    print "-- containEventSents:", len(containEventSents)
    print containEventSents.most_common()

    negSents = []
    #negSents = [line for line in allSents  if len([line2 for line2 in sentHash.keys() if line.find(line2) >= 0]) < 0]
    #oneEventSentence = dict([(sent, events[0]) for sent, events in sentHash.items() if len(events) == 1])
    eventNumInSent = [len(events) for sent, events in sentHash.items()]
    sentNumAll = len(sentHash)
    print "## #sentences", len(allSents), "#sents-no-events", len(negSents)
    print "## #sentence-with-events", sentNumAll
    print "## total #event", len(eventArr)
    print "## #events-in-sents:",
    print Counter(eventNumInSent).most_common()
    for eventNum, sentNum in Counter(eventNumInSent).most_common():
        print eventNum, "\t", sentNum, "\t", sentNum*100.0/sentNumAll

    return sentHash, random.sample(negSents, min(negSentNum, len(negSents)))

def extractSpan_word(words, wordsArg):
    strWhole = " ".join(words)
    strPart = " ".join(wordsArg)
    #if len(re.findall(strPart, strWhole)) > 1:
    #    print "-- Multi-instance argument", strPart, strWhole
    index = strWhole.find(strPart)
    preWordNum = len(strWhole[:index].strip().split())
    return (preWordNum, preWordNum+len(wordsArg)-1)

def negSent2JointTrain(negSents, posSentNum):
    neg_training_data = []
    for sentId, sent in enumerate(negSents):
        wordsIn = wordpunct_tokenize(sent)
        eventTypeSequence = [-1 for i in range(len(wordsIn))]
        neg_training_data.append((str(sentId + posSentNum), sent, eventTypeSequence))
    return neg_training_data

# -InputFormat of sentHash: sent:eventArr
#   -event = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
# -OutputFormat of jointTrain: (sentenceId, sentence, eventTypeSequence, eventArr)
#   -event: triggerIndex, roleSequence
def sent2JointTrain(sentHash, testFlag, eventSubTypeRoleHash):
    #eventTypeArr = sorted(eventSubTypeRoleHash.keys())

    debug = False
    training_data = []
    sentenceArr = []
    for sentenceRaw, eventArr in sentHash.items():
        sentenceArr.append(sentenceRaw)
        sentId = len(sentenceArr) - 1

        sentenceRaw = sentenceRaw.replace("&", "&amp;")
        wordsIn = wordpunct_tokenize(sentenceRaw)
        sentenceText = " ".join(wordsIn)
        if debug:
            print "--------------------------------"
            print "-- raw sent:", sentenceRaw
            print "-- wordsInSent:", wordsIn

        eventTypeSequence = [-1 for i in range(len(wordsIn))]
        trigEventArr = []

        for event in eventArr:
            sentence, _, eventSubType, trigger = event[:4]
            if len(sentence[0]) != (sentence[1][1]-sentence[1][0]+1):
                print "-- Warning!! sentence changed:", len(sentence[0]), Tab, sentence[1], Tab, sentence[0]
            eventSubType = eventSubType.lower()
            args = event[4:]

            #roleSet = eventSubTypeRoleHash[eventSubType]

            # triggerText, triggerIndex
            # event type sequence
            trigger_text, trg_index = trigger
            if trigger_text == "e war":
                print "Error trigger", trigger_text
                continue
            trig_text_pre = sentenceRaw[:trg_index[0]]
            trig_text_by_index = sentenceRaw[trg_index[0]:trg_index[1]+1]
            if trigger_text != trig_text_by_index:
                print "-- Error!! trigger text, trigger char index", trigger, "###", trig_text_by_index
                continue
            wordsInTrg = wordpunct_tokenize(trigger_text)
            word_num_pre = len(wordpunct_tokenize(trig_text_pre))
            word_num_trig = len(wordsInTrg)
            trigger_index = (word_num_pre, word_num_pre+word_num_trig-1)
            #print "-- text pre, trig text", trig_text_pre, "###",  trigger_text
            #print "-- trigger raw, words, word_index", trigger, wordsIn[trigger_index[0]:trigger_index[1]+1], trigger_index
            if len(wordsInTrg) != 1:
                print "-- Multi-word trigger:", trigger_text, trigger
            if len(trigger_index) == 0:
                print "-- zero-instance trigger:", trigger_text, trigger, trigger_index
                continue
            for ti in range(trigger_index[0], trigger_index[1]+1):
                eventTypeSequence[ti] = eventSubType
            if debug:
                print "-- trigger detail:", wordsInTrg, Tab, trigger_index

            # argText, argIndex
            # role sequence
            roleSequence = ["O" for i in range(len(wordsIn))]
            for arg, role, arg_index in args:
                role = role.lower()
                wordsInArg = wordpunct_tokenize(arg)
                arg_text_pre = sentenceRaw[:arg_index[0]]
                arg_text_by_index = sentenceRaw[arg_index[0]:arg_index[1]+1]
                if arg != arg_text_by_index:
                    print "-- Error!! arg text, arg char index", (arg, role, arg_index), "###", arg_text_by_index
                    continue
                word_num_pre_arg = len(wordpunct_tokenize(arg_text_pre))
                argSpan = (word_num_pre_arg, word_num_pre_arg+len(wordsInArg)-1)
                if 0:
                    argSpan1 = extractSpan_word(wordsIn, wordsInArg)
                    if argSpan != argSpan1:
                        print "-- Multi-instance arg:", argSpan, Tab, argSpan1
                for idx in range(argSpan[0], argSpan[1]+1):
                    role_label = "I-"+role
                    if idx == argSpan[0]: role_label = "B-"+role
                    if roleSequence[idx] == "O":
                        roleSequence[idx] = role_label
                    else:
                        print "-- overlap arg", idx, Tab, wordsIn[idx], Tab, roleSequence[idx],Tab, role_label
                        roleSequence[idx] += "#"+role_label
                if debug:
                    print "-- arg detail:", role, Tab, wordsInArg, Tab, argSpan,
                    print roleSequence
            trigEventArr.append((trigger_index, roleSequence))

        #if testFlag: continue
        if debug:
            print "-- event type seq:", eventTypeSequence
            print "-- trigger events:"
            for trigIndex, roleSeq in trigEventArr:
                print "-", trigIndex, roleSeq
        training_data.append((sentId, sentenceText, eventTypeSequence, trigEventArr))
    return training_data

def obtainAllSents(dataDir):
    debug = False
    fileList = sorted(os.listdir(dataDir))
    allEvents = []
    allSents = []

    for filename in fileList:
        if debug:
            print "## Processing ", filename
        if filename.endswith(".sgm"):
            sgmContent = open(dataDir + filename, "r").readlines()
            if debug:
                print "".join(sgmContent)
            textBeginLine = [lineIdx for lineIdx, line in enumerate(sgmContent) if line.startswith("<TEXT>")]
            sgmTextContent = "".join(sgmContent[textBeginLine[0]+1:-3]).split("\n\n")
            sents = [line.replace("\n", " ").strip().rstrip(".") for line in sgmTextContent if len(line) > 1]
            if debug:
                print "---sents"
                for line in sents:
                    print line
            allSents.extend(sents)

        if not (os.path.isfile(dataDir + filename) and filename.endswith(".ee")): continue
        # from cPickle
        infile = file(dataDir+filename, "r")
        eventArrOneDoc = cPickle.load(infile)

        allEvents.extend(eventArrOneDoc)
    return allSents, allEvents


def loadArguments(filename):
    eventSubTypeHash, eventSubTypeRoleHash = loadEventHierarchy(filename)
    return set([role.lower() for argRoles in eventSubTypeRoleHash.values() for role in argRoles])

if __name__ == "__main__":
    print "Usage: python .py dataDir argumentsFile outputFilename (.../train.jointEventSent.txt)"
    print sys.argv

    testFlag = False
    if sys.argv[3].find("test.jointEventSent") >= 0: testFlag=True
    devFlag = False
    if sys.argv[3].find("dev.jointEventSent") >= 0: devFlag=True

    eventSubTypeHash, eventSubTypeRoleHash = loadEventHierarchy(sys.argv[2])
    print eventSubTypeRoleHash.items()
    Arguments = set([role.lower() for argRoles in eventSubTypeRoleHash.values() for role in argRoles])
    print "## #Arguments", len(Arguments), Arguments


    dataDir = sys.argv[1]
    allSents, allEvents = obtainAllSents(dataDir)
    #allSents = list(set(allSents))


    negSentNum = 2000
    if devFlag: negSentNum = len(allSents)
    if testFlag: negSentNum = len(allSents)


    sentHash, negSents = indexEventBySentence(allEvents, allSents, negSentNum)
    training_data = sent2JointTrain(sentHash, testFlag, eventSubTypeRoleHash)
    neg_training_data = negSent2JointTrain(negSents, len(training_data))


    if not testFlag:
        training_data.extend(neg_training_data)
    outfile = open(sys.argv[3], "w")
    cPickle.dump(training_data, outfile)
    #for jointTrain in training_data:
    #    jointTrain_str = jointTrain2str(jointTrain, " ||| ", " ")
    #    outfile.write(jointTrain_str + "\n")
    outfile.close()
    print "## events writen to", outfile.name

