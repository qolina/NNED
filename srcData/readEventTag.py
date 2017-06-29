import os
import re
import sys
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

    print event2str(eventArrOneDoc[0], "\t")
    return eventArrOneDoc

# event = [eventType, eventSubType, sentence_ldc_scope, anchor, (arg, role), (arg, role), ...]
# output: sentence[sep]eventType[sep]eventSubtype[sep]anchor[sep]arg[sep]role[sep]arg[sep]role......
def event2str(event, separator):
    (eventType, eventSubType, sentence_ldc_scope, anchor) = event[:4]
    arguments = [arg[0]+separator+arg[1] for arg in event[4:]]

    newArrangedEvent = [sentence_ldc_scope, eventType, eventSubType, anchor]
    newArrangedEvent.extend(arguments)
    eventString = separator.join(newArrangedEvent)
    #print eventString
    return eventString

# forth_layer(event): [optional]event_argument, event_mention
# fifth_layer(event_mention): extent, ldc_scope, anchor, [optional]event_mention_argument
# sixth_layer(event_mention_argument): extent
# event = [eventType, eventSubType, sentence_ldc_scope, anchor, (arg, role), (arg, role), ...]
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
        sentence_ldc_scope = eventMention[1][0].text
        sentence_ldc_scope = re.sub(r"\n", " ", sentence_ldc_scope).strip() + "."
        anchor = eventMention[2][0].text
        anchor = re.sub(r"\n", " ", anchor)
        #print "----Sentence", sentence_ldc_scope
        #print "----Anchor", anchor
        event = [eventType, eventSubType, sentence_ldc_scope, anchor]

        for eventMentionArgument in eventMention:
            if eventMentionArgument.tag != "event_mention_argument": continue
            argRole = eventMentionArgument.attrib["ROLE"]
            argText = eventMentionArgument[0][0].text
            argText = re.sub(r"\n", " ", argText)
            arg = (argText, argRole)
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
        #outfilename = dataDir + filename.strip(".apf.xml") + ".ee"
        #outfile = open(outfilename, "w")
        #for event in eventArrOneDoc:
        #    eventString = event2str(event, "\t")
        #    outfile.write(eventString + "\n")
        #outfile.close()
        #print "## Events writen to ", outfilename
