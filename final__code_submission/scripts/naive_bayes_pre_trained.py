# ## CSCI 183 Data Science
# ### Spam Filtering for Short Messages: Naive Bayes
# #### Ryan Johnson, Grace Nguyen, and Raya Young


#!usr/bin/env/python
import sklearn as sk
import nltk
import pandas as pd
import pickle
import time
from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from importlib.machinery import SourceFileLoader

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

def main():
    CreateFeatureMatrix = SourceFileLoader("CreateFeatureMatrix", "scripts/CreateFeatureMatrix.py").load_module()
    print("\n\n\n=========================== Naive Bayes ===========================")
    print("...Getting dataset")
    from CreateFeatureMatrix import getset
    training_set = CreateFeatureMatrix.getset()

    #half of the dataset will be used for training and the other half will be used for testing
    length = len(training_set)/2
    test_set = training_set[:int(length)]

    print("...Setting up vocab list")
    vocab = set(chain(*[word_tokenize(i[0].lower()) for i in training_set]))

    print("...Loading classifier")
    f = open('scripts/nb_classifier.pickle', 'rb')
    classifier = pickle.load(f)
    f.close()

    total = 0
    correct = 0

    #format of results.csv will be 2 columns, one for actual, one for predicted
    #1 means it is spam, 0 means it is not
    print("...Classifying messages (please be patient. This may take a while)")
    results = open("scripts/outputs/nb_outputs.csv", "w")
    for sentence in test_set:
        pred = 0
        act = 0
        test_sentence = clean(sentence[0])
        test_set_feat = {i:(i in word_tokenize(test_sentence.lower())) for i in vocab}
        answer = classifier.classify(test_set_feat)
        if answer == sentence[1]:
            correct = correct + 1
            is_corr = 1
    
        if answer == "spam":
            pred = 1
        if sentence[1] == "spam":
            act = 1

        results.write(str(act) + "," + str(pred) + "\n")
        total = total + 1

    classify_time = end - start
    classify_time = classify_time/60
    print("\nCorrect: " + str(correct) + ", Total: " + str(total) + "...... Time elapsed: " + str(classify_time))
    print("Naive Bayes Accuracy = ", float(correct)/float(total)*100)

    print("\n*************** Full results for Naive Bayes can be seen in nb_output.csv *****************")

if __name__ == '__main__':
    main()
