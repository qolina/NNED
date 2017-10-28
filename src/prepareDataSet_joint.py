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
def indexEventBySentence(eventArr):
    debug = False
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

    #oneEventSentence = dict([(sent, events[0]) for sent, events in sentHash.items() if len(events) == 1])
    eventNumInSent = [len(events) for sent, events in sentHash.items()]
    sentNumAll = len(sentHash)
    print "## #sentence-with-events", sentNumAll
    print "## total #event", len(eventArr)
    print "## #events-in-sents:",
    print Counter(eventNumInSent).most_common()
    for eventNum, sentNum in Counter(eventNumInSent).most_common():
        print eventNum, "\t", sentNum, "\t", sentNum*100.0/sentNumAll
    return sentHash

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
    for sentId, (sent_id, sent) in enumerate(negSents):
        wordsIn = wordpunct_tokenize(sent)
        sent = " ".join(wordsIn)
        eventTypeSequence = ["none" for i in range(len(wordsIn))]
        neg_training_data.append((str(sentId + posSentNum), sent, eventTypeSequence))
    return neg_training_data

def roleModify(eventSubType, role):
    if eventSubType in ["injure", "die"]:
        role = role.replace("victim", "person")
    elif eventSubType in ["transfer-ownership"]:
        role = role.replace("recipient", "buyer")
        role = role.replace("giver", "seller")
    elif eventSubType in ["demonstrate"]:
        role = role.replace("entity", "attacker")
    if eventSubType in ["arrest-jail", "release-parole", "execute", "extradite"]:
        role = role.replace("person", "defendant")
    if eventSubType == "release-parole":
        role = role.replace("entity", "adjudicator")
    if eventSubType == "fine":
        role = role.replace("entity", "defendant")
    if eventSubType == "extradite":
        role = role.replace("destination", "place")
        role = role.replace("origin", "place")

    return role

# -InputFormat of sentHash: sent:eventArr
#   -event = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
#   -use_eventType: whether do 8-class classification or 34-class classification
# -OutputFormat of jointTrain: (sentenceId, sentence, eventTypeSequence, event)
#   -event: triggerIndex, roleSequence
def sent2JointTrain(sentHash, testFlag, eventSubTypeRoleHash, use_eventType = False):

    #eventTypeArr = sorted(eventSubTypeRoleHash.keys())

    debug = True
    training_data = []
    sentenceArr = []
    for sentenceRaw, eventArr in sentHash.items():
        sentenceArr.append(sentenceRaw)
        sentId = len(sentenceArr) - 1

        #sentenceRaw = sentenceRaw.replace("&", "&amp;")
        wordsIn = wordpunct_tokenize(sentenceRaw)
        sentenceText = " ".join(wordsIn)
        if debug:
            print "--------------------------------"
            print "-- raw sent:", sentenceRaw
            print "-- wordsInSent:", wordsIn

        eventTypeSequence = ["none" for i in range(len(wordsIn))]
        trigEventArr = []

        for event in eventArr:
            sentence, eventType, eventSubType, trigger = event[:4]
            sent_st, sent_ed = sentence[1]
            if len(sentence[0]) != (sentence[1][1]-sentence[1][0]+1):
                print "-- Warning!! sentence changed:", len(sentence[0]), Tab, sentence[1], Tab, sentence[0]
            eventType = eventType.lower()
            eventSubType = eventSubType.lower()
            args = event[4:]

            #roleSet = eventSubTypeRoleHash[eventSubType]

            # triggerText, triggerIndex
            # event type sequence
            trigger_text, trg_index = trigger
            trg_index = (trg_index[0]-sent_st, trg_index[1]-sent_st)
            if trigger_text == "e war":
                print "Error trigger", trigger_text
                continue
            if trigger_text == "Q&A":
                trigger_text = "Q&----A"
            trig_text_pre = sentenceRaw[:trg_index[0]]
            trig_text_by_index = sentenceRaw[trg_index[0]:trg_index[1]+1]
            if trigger_text != trig_text_by_index:
                print "-- Error!! trigger text, trigger char index", trigger, "###", trig_text_by_index, trg_index
                print sentence[1], sentenceRaw
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
                if use_eventType: eventTypeSequence[ti] = eventType
                else: eventTypeSequence[ti] = eventSubType
            if debug:
                print "-- trigger detail:", wordsInTrg, Tab, trigger_index

            # argText, argIndex
            # role sequence
            roleSequence = ["O" for i in range(len(wordsIn))]
            for arg, role, arg_index in args:
                arg_index = (arg_index[0]-sent_st, arg_index[1]-sent_st)
                role = role.lower()
                # change role into first layer role
                if use_eventType: role = roleModify(eventSubType, role)
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
                        #print "-- overlap arg", idx, Tab, wordsIn[idx], Tab, roleSequence[idx],Tab, role_label
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
        if len(sentenceText.split()) != len(eventTypeSequence):
            print "## Error generated data, different length:", len(sentenceText.split()), len(eventTypeSequence)
        training_data.append((sentId, sentenceText, eventTypeSequence, trigEventArr))
    return training_data

def spanInclude(event_span, sent_span):
    (st, ed) = sent_span
    (est, eed) = event_span
    if est >= st and eed <= ed: return 1 # event is included in
    elif est >= st and est < ed and eed > ed: return 2 # event is partially included in
    else: return 0

def updateSents(sent_spans, event_span):
    debug = False
    related_sents = [(sent_id, sent_span, spanInclude(event_span, sent_span)) for sent_id, sent_span in enumerate(sent_spans) if spanInclude(event_span, sent_span) != 0]
    if debug:
        print related_sents
    if len(related_sents) != 1:
        print "##-- Error matching sent", related_sents
        return sent_spans
    if related_sents[0][2] == 2:
        sent_id = related_sents[0][0]
        sent_span = related_sents[0][1]
        j = 1
        while j < len(sent_spans)-1-sent_id:
            new_ed = sent_spans[sent_id+j][1]
            if new_ed >= event_span[1]:
                sent_spans[sent_id] = (sent_span[0], sent_spans[sent_id+j][1])
                break
            j += 1
        if debug:
            print sent_spans[sent_id:sent_id+j+1]
        for k in range(1, j+1):
            if debug:
                print "--to del", sent_spans[sent_id+1]
            del sent_spans[sent_id+1]
    return sent_spans

def rearrangeSents(content, sentences_in_doc, events_in_doc):
    debug = False
    ldc_scopes = [item[0][1] for item in events_in_doc]
    sent_spans = [item[0] for item in sentences_in_doc]
    if debug:
        print "--", sent_spans
    for event_id, ldc_scope in enumerate(ldc_scopes):
        if debug:
            print "--ldc_scope", ldc_scope, events_in_doc[event_id][0][0]
        sent_spans = updateSents(sent_spans, ldc_scope)
    new_sents = [((st, ed), content[st:ed+1]) for (st, ed) in sent_spans]
    return new_sents

# event_v2 = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
def attachSent2Events(content, sentences_in_doc, events_in_doc):

    debug = False
    matched_sents = set()
    new_events = []
    if debug:
        print "Sentences:"
        for item in sentences_in_doc:
            print item[0], item[1]
    sentences_in_doc = rearrangeSents(content, sentences_in_doc, events_in_doc)
    for event_item in events_in_doc:
        matched_sent_id = []
        ldc_text, ldc_index = event_item[0]

        for sent_id, sent_item in enumerate(sentences_in_doc):
            sent_index, sent_text = sent_item
            if ldc_index[0] >= sent_index[0] and ldc_index[1] <= sent_index[1]:
                matched_sent_id.append(sent_id)

        if len(matched_sent_id) != 1:
            print "##Error matching sents to event", ldc_index, ldc_text
            print matched_sent_id
            continue
        matched_sents.add(matched_sent_id[0])
        matched_sent = sentences_in_doc[matched_sent_id[0]]
        event_item[0] = (matched_sent[1], matched_sent[0])
        new_events.append(event_item)
    unmatched_sents = [sentences_in_doc[i] for i in range(len(sentences_in_doc)) if i not in matched_sents]
    return new_events, unmatched_sents

def obtainAllEvents(dataDir):
    debug = False
    fileList = sorted(os.listdir(dataDir))

    all_events = []
    all_neg_sents = []

    for filename in fileList:
        if not (os.path.isfile(dataDir + filename) and filename.endswith("ee")): continue
        #if not filename.startswith("CNN_CF_20030304.1900.04"): continue
        #if not filename.startswith("BACONSREBELLION_20050226.1317"): continue
        #if not filename.startswith("AGGRESSIVEVOICEDAILY_20041208.2133"): continue
        if debug:
            print "## Processing ", filename
            print "## Loading done", len(sentences_in_doc), len(eventArrOneDoc)
        # from cPickle
        infile = file(dataDir+filename, "r")
        content = cPickle.load(infile)
        sentences_in_doc = cPickle.load(infile)
        eventArrOneDoc = cPickle.load(infile)

        events_in_doc, unmatched_sents = attachSent2Events(content, sentences_in_doc, eventArrOneDoc)
        all_neg_sents.extend(unmatched_sents)
        all_events.extend(events_in_doc)
    return all_neg_sents, all_events

# -InputFormat of jointTrain: (sentenceId, sentence, eventTypeSequence, event)
#   -event: triggerIndex, roleSequence
# -OutputFormat of trigger: (sentenceId, sentence, eventTypeSequence, event)
def outputTriggerStr(event_item):
    sent_id, sent_text, event_type_seq = event_item[:3]
    #event_type_seq = [str(item) for item in event_type_seq]
    trigger_str = sent_text + "\t" + " ".join(event_type_seq)
    return trigger_str


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
    all_neg_sents, all_events = obtainAllEvents(dataDir)

    #sys.exit(0)

    negSentNum = 1000
    if devFlag or testFlag: negSentNum = len(all_neg_sents)


    sentHash = indexEventBySentence(all_events)
    training_data = sent2JointTrain(sentHash, testFlag, eventSubTypeRoleHash)
    neg_training_data = negSent2JointTrain(all_neg_sents, len(training_data))
    print "# data", len(training_data), len(neg_training_data)


    if not (devFlag and testFlag):
        neg_training_data = neg_training_data[:negSentNum]
    training_data.extend(neg_training_data)
    outfile = open(sys.argv[3], "w")
    #cPickle.dump(training_data, outfile)
    for jointTrain in training_data:
        #jointTrain_str = jointTrain2str(jointTrain, " ||| ", " ")
        #outfile.write(jointTrain_str + "\n")
        trigger_str = outputTriggerStr(jointTrain)
        outfile.write(trigger_str + "\n")
    outfile.close()
    print "## events writen to", outfile.name

