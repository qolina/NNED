import os
import re
import sys
from collections import Counter
from readEventTag import event2str

# event = [sentence_ldc_scope, eventType, eventSubType, anchor, (arg, role), (arg, role), ...]
# eventString: sentence[sep]eventType[sep]eventSubType[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
def str2event(eventString, separator):
    arr = eventString.lower().split(separator)
    arguments = arr[4:]
    arguments = [(arguments[i*2], arguments[i*2+1]) for i in range(len(arguments)/2)]

    event = arr[:4]
    event.extend(arguments)
    return event

def indexEventBySentence(eventArr):
    sentHash = {}
    for event in eventArr:
        sentence = event[0]
        if sentence in sentHash:
            sentHash[sentence].append(event)
        else:
            sentHash[sentence] = [event]

    oneEventSentence = dict([(sent, events[0]) for sent, events in sentHash.items() if len(events) == 1])
    eventNumInSent = [len(events) for sent, events in sentHash.items()]
    sentNumAll = len(sentHash)
    print "## ori #event", len(eventArr)
    print "## #sentence", sentNumAll
    print "## #events-in-sents:",
    print Counter(eventNumInSent).most_common()
    for eventNum, sentNum in Counter(eventNumInSent).most_common():
        print eventNum, "\t", sentNum, "\t", sentNum*100.0/sentNumAll
    return oneEventSentence

def event2TrainFormatStr_noType(event, sepFst, sepSnd, Arguments):
    roles_Pos = [arg[1] for arg in event[4:]]
    arguments_Pos = sepSnd.join([arg[0]+sepSnd+arg[1] for arg in event[4:]])
    # add trigger as extra argument
    arguments_Pos = event[3]+sepSnd+"trigger"+sepSnd+arguments_Pos
    # negtive argument instance
    roles_Neg = list(Arguments-set(roles_Pos))
    arguments_Neg = sepSnd.join(roles_Neg)
    return sepFst.join([event[0], arguments_Pos, arguments_Neg])

def loadArguments(filename):
    eventSubTypeHash, eventSubTypeRoleHash = loadEventHierarchy(filename)
    return set([role for argRoles in eventSubTypeRoleHash.values() for role in argRoles])

def loadEventHierarchy(filename):
    eventSubTypeRoleHash = {}
    eventSubTypeHash = {}
    content = open(filename, "r").readlines()
    eventStructures = "".join(content).split("\n\n")[1:]
    for eventStructure in eventStructures:
        if len(eventStructure.strip())==0: continue
        arr = eventStructure.strip().split("\n")
        eventType = arr[0]
        #print eventType
        for line in arr[1:]:
            eventSubType = line[:line.find(":")]
            argRoles = line[line.find(":")+1:].split()
            #print (eventSubType, argRoles)
            eventSubTypeRoleHash[eventSubType] = argRoles
            eventSubTypeHash[eventSubType] = eventType
    #print len(eventSubTypeHash)
    return eventSubTypeHash, eventSubTypeRoleHash

if __name__ == "__main__":
    print "Usage: python .py dataDir argumentsFile"
    print sys.argv

    Arguments = loadArguments(sys.argv[2])
    print "## #Arguments", len(Arguments), Arguments

    dataDir = sys.argv[1]
    fileList = sorted(os.listdir(dataDir))
    allEvents = []
    for filename in fileList:
        if not (os.path.isfile(dataDir + filename) and filename.endswith(".ee")): continue
        #print "## Processing ", filename
        content = open(dataDir+filename, "r").readlines()
        eventArrOneDoc = [str2event(line.strip(), "\t") for line in content]
        allEvents.extend(eventArrOneDoc)

    oneEventSentence = indexEventBySentence(allEvents)
    outfile = open(dataDir + "../train.oneEventSent.txt", "w")
    for sent, event in oneEventSentence.items():
        eventString = event2TrainFormatStr_noType(event, "|||", "\t", Arguments)
        outfile.write(eventString + "\n")
    outfile.close()
