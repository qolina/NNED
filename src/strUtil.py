from nltk.tokenize import sent_tokenize, word_tokenize, wordpunct_tokenize

def locateWord(word, wordsArr):
    if word in wordsArr:
        return wordsArr.index(word)
    else:
        idxs = [wordsArr.index(w) for w in wordsArr if word in wordpunct_tokenize(w)]
        return idxs[0]

def locateWords(lookupArr, wordsArr):
    if len(lookupArr) == 1:
