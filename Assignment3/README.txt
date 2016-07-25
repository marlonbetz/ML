the python file main.py should print roughtly the same results as below. Be aware that when running the script, training the convnet takes by far most of the time. 

I also added a png file with plots of the performances of all models over all folds. 
(accuracies.png).

glove.6B.50d_onlyCorpusData.txt.zip contains all embeddings for the words found in the corpus. The original txt file from the glove website was simply too big for github.




Performance of the respective models:



LOG REGRESSION

log_scores_uni [0.45000000000000001, 0.47499999999999998, 0.5, 0.505, 0.42999999999999999, 0.47499999999999998, 0.46000000000000002, 0.47999999999999998, 0.5, 0.495]
log_scores_uni_mean 0.477

log_scores_bi [0.755, 0.80500000000000005, 0.76500000000000001, 0.80000000000000004, 0.73999999999999999, 0.78000000000000003, 0.77500000000000002, 0.76500000000000001, 0.81999999999999995, 0.82999999999999996]
log_scores_bi_mean 0.7835



MLP

mlp_scores_uni [0.66500000000000004, 0.71499999999999997, 0.67500000000000004, 0.68500000000000005, 0.745, 0.69499999999999995, 0.67000000000000004, 0.68500000000000005, 0.755, 0.68999999999999995]
mlp_scores_uni_mean 0.698

mlp_scores_bi [0.72499999999999998, 0.70999999999999996, 0.68999999999999995, 0.70999999999999996, 0.72999999999999998, 0.66000000000000003, 0.63500000000000001, 0.69999999999999996, 0.745, 0.76000000000000001]
mlp_scores_bi_mean 0.7065


CONVNET

scores_conv [0.45500000000000002, 0.58999999999999997, 0.55500000000000005, 0.58499999999999996, 0.47499999999999998, 0.58999999999999997, 0.63, 0.505, 0.54000000000000004, 0.51000000000000001]
scores_conv_mean 0.5435




4.
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

In terms of complexity, the log regression of course is the best model and the MLPs the worst, given their respective number of parameters to estimate. But the Convnet needs much more time to converge, as the actual computation of gradients is much more complex.