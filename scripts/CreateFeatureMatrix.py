
# coding: utf-8

# ## CSCI 183 Data Science
# ### Spam Filtering for Short Messages: Features Matrix
# #### Ryan Johnson, Grace Nguyen, and Raya Young
# 
# 
# 

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer


# #### Import the test data 
# Separate into two arrays: spam and ham. These arrays will be processed individually in order to generate word clouds.

# In[3]:

data = pd.read_csv("../training-data/spamcollectiondata.tsv", sep='\t', names = ["Category", "Message"])
data.head()


# #### Converting to lowercase

# In[4]:

message_data = [word.lower() for word in data['Message']]
category = data['Category'].tolist()


# #### Stopword removal and Stemming
# Clean both sets of data by removing stopwords. This way, the word cloud will not be completely populated by common stop words. Stemming is also important to ensure eliminate the possibility of having multiple different forms of words.
# 

# In[5]:

stop = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')
training_set = []
i = 0
for message in message_data:
    sentence = message.split(" ")
    filtered = []
    pr = []
    for word in sentence:
        if word.lower() not in stop:
            stemmed = stemmer.stem(word)
            filtered.append(stemmed)
    pr.append(filtered)
    pr.append(category[i])
    training_set.append(pr)
    i = i+1
    
#print first 10 elements in training_set to see format
print(training_set[:10])

# #### Convert training_set to Dictionary
# 
# NLTK's classifiers only accept texts in hashable formats, so we need to convert our list into a dictionary

# In[7]:

def list_to_dict(words_list):
    return dict([(word, True) for word in words_list])
 
def getset():
    training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in training_set]
    return training_set_formatted


# In[ ]:



