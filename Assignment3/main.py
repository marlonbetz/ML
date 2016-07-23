import numpy as np 
from nltk import ngrams, FreqDist
import pickle
import sys
from os import listdir
import codecs


min_n = 5



# load texts
pathToCorpus_pos = "review_polarity/txt_sentoken/pos/"
pathToCorpus_neg = "review_polarity/txt_sentoken/neg/"

print("reading corpus ...")
texts = [codecs.open(pathToCorpus_pos+file,"r", encoding="utf-8").read() for file in listdir(pathToCorpus_pos)]
texts.extend([codecs.open(pathToCorpus_neg+file,"r", encoding="utf-8").read() for file in listdir(pathToCorpus_neg)])
# transform list into np array for multiple indexing
texts = np.array(texts)




from keras.models import Model
from keras.layers import Dense, Input, Convolution2D, MaxPooling2D,Reshape,Dropout

from sklearn.cross_validation import KFold
n_folds = 10



kf = KFold(n=len(texts), n_folds=n_folds, shuffle=True, random_state=None)

""""
1.
LOG REGRESSION
"""
log_scores_uni = []
log_scores_bi = []
c_fold = 1
for train,test in kf:
    print("cross validation fold",c_fold)
    unigrams = set()
    bigrams = set()
    freq = dict()
    print("training data size :",train.shape)
    print("test data size :",test.shape)
    """
    For symbolic ngram feature vectors as such used for the log regression here, 
    I only use ngrams found in training data as you should not know which ngrams you will see in the test set.
    """ 
    print("counting ngrams in whole corpus ....")
    for t in texts[train]:
             
        for unigram in zip(t.split()):
            unigram = tuple(unigram)
            bigrams.add(unigram)
            if unigram not in freq:
                freq[unigram] = 0
            freq[unigram] += 1
        for bigram in zip(t.split(),t.split()[1:]):
            bigrams.add(bigram)
            if bigram not in freq:
                freq[bigram] = 0
            freq[bigram] += 1
         
    #only use ngrams with minimal freq
    print("discarting ngrams under minimal freq",min_n)
    unigrams = set([ngram for ngram in unigrams if freq[ngram] >= min_n])
    bigrams = set([ngram for ngram in bigrams if freq[ngram] >= min_n])
    #dicts to convert ngrams to indices and vice versa
    print("creating ngram dicts")
    unigrams_indices = dict((c, i) for i, c in enumerate(unigrams))
    indices_unigrams = dict((i, c) for i, c in enumerate(unigrams))
    bigrams_indices = dict((c, i) for i, c in enumerate(bigrams))
    indices_bigrams = dict((i, c) for i, c in enumerate(bigrams))
         
    #initialize feature vectors with zeros everywhere
    X_ngramsRawCount_uni = np.zeros((len(texts),len(unigrams)))
    X_ngramsRawCount_bi = np.zeros((len(texts),len(bigrams)))
    y = np.zeros((len(texts), 1))
         
    #create raw freqs for log regression features
    print("creating features for log regression ...")
    for i_t,t in enumerate(texts):
        if i_t % 100 == 0:
            print(i_t, "/", len(texts),"texts processed in fold",c_fold)     
            
        #nltk freqdist to count ngrams in the texts. works faster than looping over it by hand.
        fd_uni = FreqDist(t.split())
        fd_bi = FreqDist(zip(t.split(),t.split()[1:]))
        """
        loop over all ngrams in the texts. if they occur in the ngram sets of the test set 
        (i.e. they occur at least n times), their count in the text is used as a feature.
        otherwise the respective feature value stays zero.
        """
        for unigram in fd_uni:
            if unigram in unigrams:
                X_ngramsRawCount_uni[i_t,unigrams_indices[unigram]] = fd_uni[ngram]
        for bigram in fd_bi:
            if bigram in bigrams:
                X_ngramsRawCount_bi[i_t,bigrams_indices[bigram]] = fd_bi[bigram]
  
        """
        the actual list of texts are ordered, where the first half is positive and the second half negative.
        """
        if i_t < 1000:
            y[i_t] = [True]
        else:
            y[i_t] = [False]
    """
    LOG REGRESSION MODELS
    I implemented it in keras. 
    they are all compiled anew for every fold so they do not use the weights learned in the previous fold
    (as far as I see you cannot set back weights in keras .... 
    I think I could directly set the weights for every layer with random numpy arrays instead,
     but I think recompilation of the models is just less code to write here)
    """
    input_flat_uni = Input((X_ngramsRawCount_uni.shape[1],)) 
    y_logReg_uni = Dense(1,activation="sigmoid")(input_flat_uni)
    logReg_uni = Model(input_flat_uni,y_logReg_uni)
    logReg_uni.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
    logReg_uni.fit(X_ngramsRawCount_uni[train],y[train],verbose=False)
    score = logReg_uni.evaluate(X_ngramsRawCount_uni[test],y[test],verbose=False)
    print("unigrams test loss and accuracy",score)
    log_scores_uni.append(score[1])
    input_flat_bi = Input((X_ngramsRawCount_bi.shape[1],)) 
    y_logReg_bi = Dense(1,activation="sigmoid")(input_flat_bi)
    logReg_bi = Model(input_flat_bi,y_logReg_bi)
    logReg_bi.compile("SGD", "binary_crossentropy", metrics=["accuracy"])
    logReg_bi.fit(X_ngramsRawCount_bi[train],y[train],verbose=False)
    score = logReg_bi.evaluate(X_ngramsRawCount_bi[test],y[test],verbose=False)
    print("bigrams test loss and accuracy",score)
    log_scores_bi.append(score[1])
    c_fold += 1
print("log_scores_uni",log_scores_uni)
print("log_scores_uni_mean",np.mean(log_scores_uni))
print("log_scores_bi",log_scores_bi)
print("log_scores_bi_mean",np.mean(log_scores_bi))
import matplotlib.pyplot as plt
plt.plot(log_scores_uni,color="red",label="Log Reg uni")
plt.plot(log_scores_bi,color="blue",label="Log Reg bi")
# # #plt.legend()
# # #plt.show()
# # #sys.exit()

""""
2.
MLPs with embeddings 

"""

#load embeddings (they are also used for the convnet below)
print("loading embeddings ...")
embeddings = dict()
pathToEmbeddings_zipped = "glove.6B.50d_onlyCorpusData.txt.zip"
pathToEmbeddings = "glove.6B.50d_onlyCorpusData.txt"
from zipfile import ZipFile
with ZipFile(pathToEmbeddings_zipped) as myzip:
    for line in myzip.open(pathToEmbeddings):
        l = line.split()
        key = l[0].decode("utf-8")
        vector = np.array([float(tmp) for tmp in l[1:]])
        embeddings[key] = vector
embedding_dim = 50

 
#lists to store accuracies
mlp_scores_uni = []
mlp_scores_bi = []
#initialize features as matrices of zeros
X_embeddings_sum_uni = np.zeros((len(texts), embedding_dim))
X_embeddings_sum_bi = np.zeros((len(texts), embedding_dim*2))
y = np.zeros((len(texts), 1))
  
 
         
"""
feature creation for MLPs
i first sum up all embeddings for the ngrams in the documents and than normalize them for better convergence
"""
for i_t in range(len(texts)):
    print("creating features for log regression ...")
    if i_t % 100 == 0:
        print(i_t, "/", len(texts),"documents processed")
    t = texts[i_t]
 
 
 
  
  
    # unigrams
    for token in t.lower().split():
        if token in embeddings:
            X_embeddings_sum_uni[i_t] += embeddings[token]
               
    # bigrams
    for token1, token2 in zip(t.lower().split(), t.lower().split()[1:]):
        if token1 in embeddings and token2 in embeddings:
            X_embeddings_sum_bi[i_t] += np.concatenate((embeddings[token1], embeddings[token2]))
    # assign labels ... since the first half of the languages are positive, just look for the counter as info to get label
    if i_t < 1000:
        y[i_t] = [True]
    else:
        y[i_t] = [False]
# normalize embedding data for better convergence
print("normalizing data")
X_embeddings_sum_uni = (X_embeddings_sum_uni - np.mean(X_embeddings_sum_uni)) / np.std(X_embeddings_sum_uni)
X_embeddings_sum_bi = (X_embeddings_sum_bi - np.mean(X_embeddings_sum_bi)) / np.std(X_embeddings_sum_bi)
c_fold = 1
for train, test in kf:
    print("cross validation fold",c_fold)
 
      
  
    # MLP WITH EMBEDDINGS
      
    n_hidden = 1000
    nb_epoch = 30
    # unigrams
    input_mlp_embeddings_uni = Input((embedding_dim,),name = "input_mlp_embeddings_uni")
    hidden_mlp_embeddings_uni = Dense(n_hidden, activation="relu")(input_mlp_embeddings_uni)
    y_mlp_embeddings_uni = Dense(1, activation="sigmoid")(hidden_mlp_embeddings_uni)
    mlp_embeddings_uni = Model(input_mlp_embeddings_uni, y_mlp_embeddings_uni)
    mlp_embeddings_uni.compile("Adam", "binary_crossentropy", metrics=["accuracy"])
    mlp_embeddings_uni.fit(X_embeddings_sum_uni[train], y[train], verbose="False",nb_epoch=nb_epoch)
    score = mlp_embeddings_uni.evaluate(X_embeddings_sum_uni[test], y[test])
    print("mlp unigrams test loss and accuracy",score)
    mlp_scores_uni.append(score[1])
      
    # bigrams
    input_mlp_embeddings_bi = Input((embedding_dim*2,),name="input_mlp_embeddings_bi")
    hidden_mlp_embeddings_bi = Dense(n_hidden, activation="relu")(input_mlp_embeddings_bi)
    y_mlp_embeddings_bi = Dense(1, activation="sigmoid")(hidden_mlp_embeddings_bi)
    mlp_embeddings_bi = Model(input_mlp_embeddings_bi, y_mlp_embeddings_bi)
    mlp_embeddings_bi.compile("Adam", "binary_crossentropy", metrics=["accuracy"])
    mlp_embeddings_bi.fit(X_embeddings_sum_bi[train], y[train],verbose="False",nb_epoch=nb_epoch)
    score = mlp_embeddings_bi.evaluate(X_embeddings_sum_bi[test], y[test])
    print("mlp bigrams test loss and accuracy",score)
    mlp_scores_bi.append(score[1])
    c_fold +=1
import matplotlib.pyplot as plt
print("mlp_scores_uni",mlp_scores_uni)
print("mlp_scores_uni_mean",np.mean(mlp_scores_uni))
print("mlp_scores_bi",mlp_scores_bi)
print("mlp_scores_bi_mean",np.mean(mlp_scores_bi))
plt.plot(mlp_scores_uni,color="yellow",label="MLP uni")
plt.plot(mlp_scores_bi,color="orange",label="MLP bi")
# plt.legend()
# plt.show()
# sys.exit()
"""
3.
CONVNET
FOR CONVNETS I ONLY USE THE FIRST TWO HUNDRED WORDS, OTHERWISE MY RAM IS TOTALLY CLUTTERED 
(I only have 4 GB)
"""
n_wordsPerDoc = 200

##feature generation for convnets
X_embeddings_cnn = np.zeros((len(texts), n_wordsPerDoc,embedding_dim))
y = np.zeros((len(texts), 1))

for i_t in range(len(texts)):
    if i_t % 100 == 0:
        print(i_t, "/", len(texts),"documents processed")
    t = texts[i_t]
    




    # create embedding features convnets
    # unigrams
    c_token = 0
    for token in t.lower().split()[:n_wordsPerDoc]:
        #check if token is in embedding model, otherwise leave it being zeros
        if token in embeddings:
            X_embeddings_cnn[i_t,c_token] = embeddings[token]
                 
        c_token +=1
    if i_t < 1000:
        y[i_t] = [True]
    else:
        y[i_t] = [False]
  
# normalize embedding data for better convergence
print("normalizing data")
X_embeddings_cnn = (X_embeddings_cnn - np.mean(X_embeddings_cnn)) / np.std(X_embeddings_cnn)


#train and evaluate with cv
scores_conv = []
c_fold = 1

for train, test in kf:
    print("cross validation fold",c_fold)
    """
    CONVNET
    two convlayers (4*embedding_dim kernel in the first layer, then 3*3, 5 output channels each) with following max pooling after each layer. 
    After that a fully connected layer with dropout and a binary sigmoid classifier
    """
    
    n_outputChannels = 5
    nb_epoch = 20
    p_droput = .5
    input_conv = Input((1,n_wordsPerDoc,embedding_dim))
    x = Convolution2D(n_outputChannels,4,embedding_dim,activation="relu",border_mode="same")(input_conv)
    x = MaxPooling2D((5,5))(x)
    #now its n_outputChannels*40*10
    x = Convolution2D(n_outputChannels,3,3,activation="relu",border_mode="same")(x)
    x = MaxPooling2D((2,2))(x)
    #now its n_outputChannels*20*5

    x = Reshape((n_outputChannels*20*5,))(x)
    x = Dense(100,activation="relu")(x)
    x = Dropout(p_droput)(x)
    y_conv = Dense(1,activation="sigmoid")(x)
    convnet = Model(input_conv,y_conv)
    print("compiling conv net ...")
    convnet.compile("Adam", "binary_crossentropy", metrics=["accuracy"])

    convnet.fit(X_embeddings_cnn[train].reshape(-1,1,n_wordsPerDoc,embedding_dim),y[train],nb_epoch=nb_epoch,verbose=True)
    score = convnet.evaluate(X_embeddings_cnn[test].reshape(-1,1,n_wordsPerDoc,embedding_dim),y[test],verbose=False)
    print("convnet test loss and accuracy",score)
    scores_conv.append(score[1])
import matplotlib.pyplot as plt
print("mlp_scores_uni",mlp_scores_uni)
print("mlp_scores_uni_mean",np.mean(mlp_scores_uni))
print("mlp_scores_bi",mlp_scores_bi)
print("mlp_scores_bi_mean",np.mean(mlp_scores_bi))
print("log_scores_uni",log_scores_uni)
print("log_scores_uni_mean",np.mean(log_scores_uni))
print("log_scores_bi",log_scores_bi)
print("log_scores_bi_mean",np.mean(log_scores_bi))
print("scores_conv",scores_conv)
print("scores_conv_mean",np.mean(scores_conv))
plt.plot(scores_conv,color="black",label="ConvNet")
plt.legend()
plt.show()

"""
For the log regression, the poor performance for the unigram model comes from unigrams being poor predictors in general as they do not capture the context of the ngrams.
The bigram model performs better since it captures some syntactic structure of the documents. In general the benefit of simple log regression is its structural restriction being a simple additive model - it cannot overfit to the training data as severely as networks with hidden layers.

Both MLPs work better than the unigram log regression, since embeddings can capture both semantic and syntactical information.
However, the order of the ngrams (i.e. the cardinality of n) does not really make a difference apparently. The embeddings are trained as unigrams and already capture their context information, 
so I honestly do not now if it makes sense to use bigrams here, 
but it might be nevertheless worth exploring bigram embeddings here (I guess you would need a giant corpus then, though).
The summation of embeddings makes sense insofar as you usually have nice linear dependencies in embedding spaces, so that 
the sum of two embeddings can be understood as the intersection of the semantic / syntactic information of the single embeddings.

The convnet - here at least  - performs bad, since it is a big architecture trained on a reasonably small data set but which is, in terms of features, quite huge, 
and hence underfits severly (i.e. you usually expect large models to overfit, but here the feature space is just to big to learn anything given the relatively small number of data points relative to the feature space). 
Also the amount of data to be stored leaves the question open if it makes sense to use convnets here.
In general Convnets are good for image processing, but sequences should be processed with recursive or recurrent networks instead.

For all three models, I did not try to make use of regularization to fine-tune the models (except for the dropout layer for the convnet), hence it can be that 
the models *can* perform better than seen here (I guess especially for the MLPs this should be the case, because of their fully connected layers they love to overfit
... on the other hand, the convnet is also quite big, so you should use some regularization here too, but then the question arrises here too if the training set is just too small for such big networks). 
Fine-tuning of NNs can take a lot of time especially which bigger NNs, which I unfortunately did not have this week 
(I have to use this laptop to do other stuff too =)). Another thing would be using a validation set, but you did not directly ask for it.

In general, using embeddings instead of traditional ngrams has the advantage that out of vocabulary items not contained in the
training set can still be processed as long as they can be embedded into the embedding space 
(you either take pretrained embeddings from giant corpora out of the box or you train domain specific embeddings 
and train a regression from one corpus into the other one, in neural SMT systems you often see that as far as I can tell) 

In terms of complexity, the log regression of course is the best model and the convnet the worst, given their respective number of parameters to estimate.
"""