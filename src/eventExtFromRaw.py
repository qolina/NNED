import os
import re
import sys
import cPickle
from aceEventUtil import event2str
from xml.etree.ElementTree import ElementTree

# root: sourcefile
# second_layer: document
# third_layer: entity, timex, relation, event
def extractEvents(filename):
    xmlTree = ElementTree(file=filename)
    #root = xmlTree.getroot()

    eventArrOneDoc = []
    for eventEle in xmlTree.iter(tag="event"):
        #print eventEle
        #print eventEle.tag, eventEle.attrib
        eventArr = extractEvent(eventEle)
        #print eventArr
        eventArrOneDoc.extend(eventArr)

    #print event2str(eventArrOneDoc[0], "\t")
    return eventArrOneDoc

# forth_layer(event): [optional]event_argument, event_mention
# fifth_layer(event_mention): extent, ldc_scope, anchor, [optional]event_mention_argument
# sixth_layer(event_mention_argument): extent
# event_v1 = [sentence_ldc_scope, eventType, eventSubType, anchorText, (argText, role), (argText, role), ...]
# event_v2 = [(sentence_ldc_scope, index), eventType, eventSubType, (anchorText, index), (argText, role, index), (argText, role, index), ...]
def extractEvent(eventEle):
    #print eventEle.attrib
    eventType = eventEle.attrib["TYPE"]
    eventSubType = eventEle.attrib["SUBTYPE"]
    #print "-- Event type, subtype", eventType, "\t",  eventSubType

    #print "-- Event Arguments:", 
    #for eventArgument in eventEle:
    #    if eventArgument.tag != "event_argument": continue
    #    print eventArgument.attrib["ROLE"], 
    #print

    eventArr = []
    #print "-- Event Mention:" 
    for eventMention in eventEle:
        if eventMention.tag != "event_mention": continue
        sentenceElement = eventMention[1][0]
        sentence_ldc_scope = sentenceElement .text
        sentence_ldc_scope = re.sub(r"\n", " ", sentence_ldc_scope).strip()
        sentence_index = (int(sentenceElement.attrib["START"]), int(sentenceElement.attrib["END"]))
        sentence = (sentence_ldc_scope, (0, sentence_index[1]-sentence_index[0]))

        anchorEle = eventMention[2][0]
        anchorText = anchorEle.text
        anchorText = re.sub(r"\n", " ", anchorText)
        anchor_index = (int(anchorEle.attrib["START"])-sentence_index[0], int(anchorEle.attrib["END"])-sentence_index[0])
        anchor = (anchorText, anchor_index)
        #print "----Sentence", sentence
        #print "----Anchor", anchor

        event = [sentence, eventType, eventSubType, anchor]

        for eventMentionArgument in eventMention:
            if eventMentionArgument.tag != "event_mention_argument": continue
            argRole = eventMentionArgument.attrib["ROLE"]
            argElement = eventMentionArgument[0][0]
            argText = argElement .text
            argText = re.sub(r"\n", " ", argText)
            arg_index = (int(argElement.attrib["START"])-sentence_index[0], int(argElement.attrib["END"])-sentence_index[0])
            arg = (argText, argRole, arg_index)
            event.append(arg)
            #print arg
        eventArr.append(event)
    return eventArr

if __name__ == "__main__":
    print "Usage: python readEventTag.py dataDir"

    dataDir = sys.argv[1]
    fileList = sorted(os.listdir(dataDir))
    for filename in fileList:
        if not (os.path.isfile(dataDir + filename) and filename.endswith(".apf.xml")): continue
        print "## Processing ", filename
        eventArrOneDoc = extractEvents(dataDir+filename)
        if len(eventArrOneDoc) == 0: continue
        outfilename = dataDir + filename.replace("apf.xml", "ee")
        outfile = open(outfilename, "w")
        cPickle.dump(eventArrOneDoc, outfile)
        ## to string version for output
        #for event in eventArrOneDoc:
        #    print event
        #    eventString = event2str(event, "|||")
        #    outfile.write(eventString + "\n")
        outfile.close()
        print "## Events writen to ", outfilename
