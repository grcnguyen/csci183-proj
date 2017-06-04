import sklearn as sk
import nltk
import pandas as pd
import numpy as np
from itertools import chain
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

data = pd.read_csv("../training-data/spamcollectiondata.tsv", sep='\t', names = ["Category", "Message"])
data.head()

d2_messages = list()
d2_cats = list()
with open("../training-data/english_big.csv", "r", errors='ignore') as test:
    lines = test.readlines()
    np.random.shuffle(lines)
    for l in lines:
        words = l.split(",")
        message = ""
        for w in words:
            if w == 'ham\n' or w=='spam\n':
                d2_messages.append(message)
                if w == 'ham\n':
                    d2_cats.append('ham')
                else:
                    d2_cats.append('spam')
            else:
                message = message + w

message_data = [word.lower() for word in data['Message']]
category = data['Category'].tolist()

message_data.extend(d2_messages)
category.extend(d2_cats)

stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
training_set = []
i = 0
for message in message_data:
    sentence = message.split(" ")
    filtered = ""
    pr = []
    for word in sentence:
        if word.lower() not in stop:
            stemmed = stemmer.stem(word)
            filtered = filtered + " " + stemmed
    pr.append(filtered)
    pr.append(category[i])
    training_set.append(pr)
    i = i+1

def getset():
    return training_set
