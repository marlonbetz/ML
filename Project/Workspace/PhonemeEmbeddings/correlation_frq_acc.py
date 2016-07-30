import numpy as np 
from pandas import DataFrame
import pickle
import codecs 
import regex
df = pickle.load(open("mean_acc_topn1.pkl","rb"))
print(df.columns)
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
for i,line in enumerate(codecs.open(pathToASJPCorpusFile,"r","utf-8")):
    if i > 0:
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
            for i_w,word in enumerate(words):
                if "," in word:
                    for match in  regex.findall("(?<=,).+",word):          
                        tba.append(match)
                    #reduce entry to first occurence of seperator
                    words[i_w] = word[:word.index(",")]
            words.extend(tba)
            allWords.extend(words)
   
"""
EXTRACT ALL PHONEMES AND ADD WORD BOUNDARIES AND GET RID OF EMPTY STRINGS
"""
print("EXTRACT ALL PHONEMES AND ADD WORD BOUNDARIES AND GET RID OF EMPTY STRINGS")
allWords = [["<s>"]+getListofASJPPhonemes(word)+["</s>"] for word in allWords if len(word) > 0]

"""
COUNT PHONEMES
"""
print("COUNT PHONEMES")
freq= dict()
tags = ["labialized","palatalized","aspirated","glottalized","plain"]
for tag in tags:
    freq[tag] = 0
for i,word in enumerate(allWords):
    for phoneme in word:
        if "w~" in phoneme:
            freq["labialized"] += 1
        if "y~" in phoneme:
            freq["palatalized"] += 1
        if "h~" in phoneme:
            freq["aspirated"] += 1
        if "\"" in phoneme:
            freq["glottalized"] += 1
        if len(phoneme) == 1 and not regex.search(".*[aeiou3].*", phoneme):
            freq["plain"] += 1

print(freq)
x = np.array([np.log(freq[tag]) for tag in tags])
y = np.array([df[tag].mean() for tag in tags])
print(x,y)
from scipy.stats import pearsonr,spearmanr
print(pearsonr(x, y))
print(spearmanr(x, y))
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x.reshape((-1,1)),y)

import matplotlib.pyplot as plt
for i,tag in enumerate(tags):
    plt.scatter(x[i],y[i])
    plt.annotate(tag,(x[i],y[i]))
plt.plot(np.arange(0,20),lm.predict(np.arange(0,20).reshape((-1,1))))
plt.xlabel("frequency in log space")
plt.ylabel("accuracy in analogy tasks")
plt.show()