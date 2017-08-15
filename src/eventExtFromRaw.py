import os
import re
import sys
import cPickle
from aceEventUtil import event2str
from xml.etree.ElementTree import ElementTree
from sgmllib import SGMLParser
from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize
import nltk.data

class aceParser(SGMLParser):
    content = ""
    line_content = []
    def init(self, path):
        self.path = path
    def handle_data(self, text):
        text = text.replace("&", "&----")
        self.line_content.append(text)
        self.content += text

def parseSGML(filename):
    debug = True
    sgm_parser = aceParser()

    file_content = open(filename, "r").read()
    sgm_parser.content = ""
    sgm_parser.line_content = []
    sgm_parser.feed(file_content)
    content = sgm_parser.content
    line_content = sgm_parser.line_content
    content = content.replace("\n", " ")
    if filename.find("FLOPPINGACES_20041114.1240.03") >= 0:
        content = content.replace("&", "&----")
        line_content = [line.replace("&----", "&") for line in line_content]
    sentences = []

    sent_detector = nltk.data.load("tokenizers/punkt/english.pickle")
# test output
    if debug:
        #idx = (1527, 1536)
        idx = (718, 909)
        print "# direct from new content:", content[idx[0]:idx[1]+1]

    sent_id = 0
    line_id = 0
    while line_id < len(line_content):
        line = line_content[line_id]
        pre_content = "".join(line_content[:line_id])
        char_st = len(pre_content)

        while line_id < len(line_content)-2 and line_content[line_id+1] == "&----":
            line = line + line_content[line_id+1] + line_content[line_id+2]
            line_id += 2
        line_id += 1
        line = line.replace("\n", " ")
        char_ed = char_st + len(line)
        if debug:
            print "-----------------------------", line_id, (char_st, char_ed)
            print "S-"+line+"-E"

        if len(line.strip())<1: continue

        #sents_in_line = line.split("\n\n")
        #print line
        sents_in_line = sent_tokenize(line)
        last_end = 0
        #sents_in_line = sent_detector.tokenize(line)
        for sent in sents_in_line:
            sent = sent.replace("\n", " ").strip()
            sent_st_in_line = line.find(sent, last_end)
            sent_ed_in_line = sent_st_in_line + len(sent) - 1
            last_end = sent_ed_in_line
            sent_st = char_st + sent_st_in_line
            sent_ed = sent_st + len(sent) - 1
            sent_id += 1
            if debug:
                print "------##", sent_id, (sent_st_in_line, sent_ed_in_line), (sent_st, sent_ed)
                print sent
            sentences.append(((sent_st, sent_ed), sent))
    #for sent_id, (sent_span, sent) in enumerate(sentences[:]):
    #    print "##",sent_id, sent_span, sent
    return sentences[3:], content
    #return line_content

# root: sourcefile  *.sgm eventfilename *.apf.xml
# second_layer: document
# third_layer: entity, timex, relation, event
def extractEvents(filename):

    xmlTree = ElementTree(file=filename[:-3]+"apf.xml")
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
        #sentence = (sentence_ldc_scope, (0, sentence_index[1]-sentence_index[0]))
        sentence = (sentence_ldc_scope, sentence_index)

        anchorEle = eventMention[2][0]
        anchorText = anchorEle.text
        anchorText = re.sub(r"\n", " ", anchorText)
        #anchor_index = (int(anchorEle.attrib["START"])-sentence_index[0], int(anchorEle.attrib["END"])-sentence_index[0])
        anchor_index = (int(anchorEle.attrib["START"]), int(anchorEle.attrib["END"]))
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
            #arg_index = (int(argElement.attrib["START"])-sentence_index[0], int(argElement.attrib["END"])-sentence_index[0])
            arg_index = (int(argElement.attrib["START"]), int(argElement.attrib["END"]))
            arg = (argText, argRole, arg_index)
            event.append(arg)
            #print arg
        eventArr.append(event)
    return eventArr

def main():
    dataDir = sys.argv[1]
    fileList = sorted(os.listdir(dataDir))
    line_num = 0
    for filename in fileList[:]:
        if not (os.path.isfile(dataDir + filename) and filename.endswith(".sgm")): continue
        sentences_in_doc, content = parseSGML(dataDir+filename)
        line_num += len(sentences_in_doc)
        eventArrOneDoc = extractEvents(dataDir+filename)
        #if len(eventArrOneDoc) == 0: continue
        outfilename = dataDir + filename[:-3]+"ee"
        outfile = open(outfilename, "w")
        cPickle.dump(content, outfile)
        cPickle.dump(sentences_in_doc, outfile)
        cPickle.dump(eventArrOneDoc, outfile)
        ## to string version for output
        #for event in eventArrOneDoc:
        #    print event
        #    eventString = event2str(event, "|||")
        #    outfile.write(eventString + "\n")
        outfile.close()
        #print "## Events writen to ", outfilename
    #print line_num
if __name__ == "__main__":
    #print "Usage: python readEventTag.py dataDir"
    main()
