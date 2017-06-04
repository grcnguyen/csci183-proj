
# coding: utf-8

# ## CSCI 183 Data Science
# ### Spam Filtering for Short Messages: Naive Bayes
# #### Ryan Johnson, Grace Nguyen, and Raya Young


#!usr/bin/env/python
import sklearn as sk
import CreateFeatureMatrix
import nltk
import pandas as pd
from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Define training function:
def clean(sent):
    clean_sent = ""
    stemmed = ""
    stop = set(stopwords.words('english'))
    stemmer = SnowballStemmer('english')
    for word in sent:
        if word not in stop:
            if isinstance(word, str):
                word.lower()
            stemmed = stemmed + word
            clean_sent = clean_sent + word
    return clean_sent

from CreateFeatureMatrix import getset
training_set = CreateFeatureMatrix.getset()

#half of the dataset will be used for training and the other half will be used for testing
length = len(training_set)/2
test_set = training_set[:int(length)]
training_set = training_set[int(length):]
#print(training_set)

vocab = set(chain(*[word_tokenize(i[0].lower()) for i in training_set]))
training_set = [({word: (word in word_tokenize(x[0])) for word in vocab}, x[1]) for x in training_set]

classifier = nltk.NaiveBayesClassifier.train(training_set)
#classifier.show_most_informative_features()

total = 0
correct = 0
for sentence in test_set:
    test_sentence = clean(sentence[0])
    test_set_feat = {i:(i in word_tokenize(test_sentence.lower())) for i in vocab}
    answer = classifier.classify(test_set_feat)
    if answer == sentence[1]:
        correct = correct + 1
    print("Answer: " + answer + ", Actual: " + sentence[1])
    total = total + 1

print("Accuracy = ", float(correct)/float(total)*100)


