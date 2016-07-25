import numpy as np 
import regex  
import codecs
import sys

def vectorLinspace(start,stop,num=50):
    assert len(start) == len(stop)
    assert num > 0
    
    return np.array([np.linspace(start[dim],stop[dim],num) for dim in range(len(start))]).transpose()

def getListofASJPPhonemes(word):
    phonemes_alone="pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieaouE3"
    phonemeSearchRegex = "["+phonemes_alone+"][\"\*]?(?!["+phonemes_alone+"]~|["+phonemes_alone+"]{2}\$)|["+phonemes_alone+"]{2}?~|["+phonemes_alone+"]{3}?\$"
    return regex.findall(phonemeSearchRegex, word)



"""
READ CORPUS FROM ASJP DUMP
"""
print("READ CORPUS FROM ASJP DUMP")
pathToASJPCorpusFile = "data/dataset.tab"
allWords = []
for line in codecs.open(pathToASJPCorpusFile,"r","utf-8"):
    line = line.split("\t")

    if "PROTO" not in line[0] and "ARTIFICIAL" not in line[2]:
        words = line[10:]
        #remove invalid characters
        for word in words:
            word = word.replace("%","")
        """
        for cells with more than one corresponding word, add that word as new entry
        """
        tba = []
        
        for word in words:
            if "," in word:
                for match in  regex.findall("(?<=,).+",word):
                    
                    tba.append(match)
                word = word[:word.index(",")]
        words.extend(tba)
        allWords.extend(words)
   
"""
EXTRACT ALL PHONEMES AND ADD WORD BOUNDARIES
"""
print("EXTRACT ALL PHONEMES AND ADD WORD BOUNDARIES")
allWords = [["<s>"]+getListofASJPPhonemes(word)+["</s>"] for word in allWords]

"""
REDUCE COMPLEX PHONEMES TO SINGLE PHONEMES IF FREQ SMALLER THAN X
"""
print("REDUCE COMPLEX PHONEMES TO SINGLE PHONEMES IF FREQ SMALLER THAN X")
minCount = 20
freq_phonemes = dict()
for i,word in enumerate(allWords):
    for phoneme in word:
        if phoneme not in freq_phonemes:
            freq_phonemes[phoneme] = 0
        freq_phonemes[phoneme] += 1
for i,word in enumerate(allWords):
    tmp = []
    for phoneme in word:
        if freq_phonemes[phoneme] >= minCount:
            tmp.append(phoneme)
        else:
            tmp.append(regex.findall("[pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieaouE3]",phoneme))
    allWords[i] = tmp
print(len(allWords))
c = 0
for word in allWords:
    c += len(word)
print(c)
sys.exit()       
from gensim.models import Word2Vec
print("fitting model")
Word2Vec(sentences=allWords)
