import numpy as np 
from pandas import DataFrame
from gensim.models import Word2Vec
import regex  
import codecs
from scipy.spatial.distance import cosine
from sklearn.neighbors import KNeighborsClassifier
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
def getListofASJPPhonemes(word):
    phonemes_alone="pbmfv84tdszcnSZCjT5kgxNqGX7hlLwyr!ieaouE3"
    phonemeSearchRegex = "["+phonemes_alone+"][\"\*]?(?!["+phonemes_alone+"]~|["+phonemes_alone+"]{2}\$)|["+phonemes_alone+"]{2}?~|["+phonemes_alone+"]{3}?\$"
    return regex.findall(phonemeSearchRegex, word)
def plot_phonemes(pathToASJPCorpusFile):
    """
    READ CORPUS FROM ASJP DUMP
    """
    print("READ CORPUS FROM ASJP DUMP")
    #pathToASJPCorpusFile = "data/dataset.tab"
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
    freq_phonemes = dict()
    for i,word in enumerate(allWords):
        for phoneme in word:
            if phoneme not in freq_phonemes:
                freq_phonemes[phoneme] = 0
            freq_phonemes[phoneme] += 1
    """
    REDUCE COMPLEX PHONEMES TO SINGLE PHONEMES IF FREQ SMALLER THAN X
    """
    
    n_ensemble  =10
    sg = 0
    hs = 1
    dim_embedding = 150
    window =1
    negative = 0
    mean_acc_all = []
    
    
    print("fitting model")
    w2v_model = Word2Vec(sentences=allWords,
                         sg = sg,
                         size=dim_embedding,
                         window=window,
                         negative=negative,
                         hs=hs,
                         min_count=1
                         )
    embeddings = dict()
    for key in w2v_model.vocab.keys():
        embeddings[key] = w2v_model[key]
    embeddings = DataFrame(embeddings,columns=embeddings.keys())
    print(embeddings.columns)
    
    m = TSNE()
    embeddings_tsne = m.fit_transform(embeddings.transpose())
    for p,emb in zip(embeddings.columns, embeddings_tsne):
            if regex.search("^.$", p):
                c = "black"
                plt.annotate(p,(emb[0],emb[1]),color=c)
            if regex.search("^[aeiou3E][*]?$", p):
                c = "red"
                plt.annotate(p,(emb[0],emb[1]),color=c)
            if regex.search("^.*w~$", p):
                c = "blue"
                plt.annotate(p,(emb[0],emb[1]),color=c)
            if regex.search("^.*y~$", p):
                c = "yellow"
                plt.annotate(p,(emb[0],emb[1]),color=c)
            if regex.search("^.*h~$", p):
                c = "brown"
                plt.annotate(p,(emb[0],emb[1]),color=c)
            if regex.search("^.*\"$", p):
                c = "green"
                plt.annotate(p,(emb[0],emb[1]),color=c)
plt.subplot(2,2,1)                
plot_phonemes("data/dataset.tab")
plt.subplot(2,2,2)                
plot_phonemes("data/dataset.tab")
plt.subplot(2,2,3)                
plot_phonemes("data/dataset.tab")
plt.subplot(2,2,4)                
plot_phonemes("data/dataset.tab")
plt.show()