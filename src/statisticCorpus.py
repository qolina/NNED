import os
import re
import sys
from collections import Counter
from readEventTag import event2str

# event = [eventType, eventSubType, sentence_ldc_scope, anchor, (arg, role), (arg, role), ...]
# eventString: sentence[sep]eventType[sep]eventSubType[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
def str2event(eventString, separator):
    arr = eventString.split(separator)
    #sentence_ldc_scope, eventType, eventSubType, anchor = arr[:4]
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

if __name__ == "__main__":
    print "Usage: python statisticCorpus.py dataDir"
    print sys.argv

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
        eventString = event2str(event, "\t")
        outfile.write(eventString + "\n")
    outfile.close()
