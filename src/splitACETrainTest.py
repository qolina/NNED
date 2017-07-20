
import os
import sys

if __name__ == "__main__":
    print "Usage: python splitTrainTest.py dataDir testFilenames"
    testFiles = open(sys.argv[2], "r").readlines()
    testFiles = [line.strip().strip(".sgm") for line in testFiles]

    dataPath = sys.argv[1]
    fileList = os.listdir(dataPath)
    for fileItem in fileList:
        if os.path.isdir(dataPath+fileItem): continue
        print "## Processing ", fileItem
        filename = fileItem.strip(".sgm").strip(".apf.xml")
        if filename in testFiles:
            os.rename(dataPath+fileItem, dataPath + "/test/"+fileItem)
        else:
            os.rename(dataPath+fileItem, dataPath + "/train/"+fileItem)
