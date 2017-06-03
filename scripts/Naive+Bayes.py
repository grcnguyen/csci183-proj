
# coding: utf-8

# ## CSCI 183 Data Science
# ### Spam Filtering for Short Messages: Naive Bayes
# #### Ryan Johnson, Grace Nguyen, and Raya Young

# In[10]:

#!usr/bin/env/python

import sklearn as sk
import CreateFeatureMatrix
import nltk
from nltk.tokenize import TweetTokenizer
from sklearn.naive_bayes import MultinomialNB

# Define training function:

# In[11]:

from CreateFeatureMatrix import getset
training_set = CreateFeatureMatrix.getset()

#tokenize set
#all_words = set(word.lower() for passage in training_set for word in word_tokenize(passage[0]))
#tokens = [({word: (word in word_tokenize(x[0])) for word in all_words}, x[1]) for x in training_set]
#print(training_set)
classifier = nltk.NaiveBayesClassifier.train(training_set)
classifier.show_most_informative_features()

tkn = TweetTokenizer()
test_sentence = "Check out this message now. is it ham is it spam? Who knows"
test_set_feat = tkn.tokenize(test_sentence)
classifier.classify(test_set_feat)
#print "%.3f" % nltk.classify.accuracy(cl, test_set)
#cl.show_most_informative_features(40)
#cl.prob_classify(featurize(name)) # get a confidence for the prediction


# In[ ]:



