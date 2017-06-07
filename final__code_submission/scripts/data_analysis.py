
# Basic
import io
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices

# SKLearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score

# NLTK
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def main(arg):
    data = pd.read_csv("../data/spamcollectiondata.tsv", sep='\t', names = ["Category", "Message"])
    if arg == "Capitals":
        # ### Comparing freq of capital letters
        ham_caps = []
        spam_caps = []

        for index,row in data.iterrows():
            if row.Category == 'ham':
                ham_caps.append(sum(1.0 for c in row.Message if c.isupper())/sum(1.0 for c in row.Message))
            else:
                spam_caps.append(sum(1.0 for c in row.Message if c.isupper())/sum(1.0 for c in row.Message))

        
        ham_weight = np.ones_like(ham_caps)/len(ham_caps)
        spam_weight = np.ones_like(spam_caps)/len(spam_caps)

        fig, ax = plt.subplots()
        ax.hist(ham_caps,weights=ham_weight,color='b',alpha=0.5,label='Ham')
        ax.hist(spam_caps,weights=spam_weight,color='g',alpha=0.5,label='Spam')
        ax.set(title='Freq. of Capital Letters')
        plt.legend()
        plt.show()

    elif arg == "Numbers":
        # ### Comparing freq of numbers
        ham_nums = []
        spam_nums = []

        for index,row in data.iterrows():
            if row.Category == 'ham':
                ham_nums.append(sum(1.0 for c in row.Message if c.isdigit())/sum(1.0 for c in row.Message))
            else:
                spam_nums.append(sum(1.0 for c in row.Message if c.isdigit())/sum(1.0 for c in row.Message))

        
        ham_weight = np.ones_like(ham_nums)/len(ham_nums)
        spam_weight = np.ones_like(spam_nums)/len(spam_nums)

        fig, ax = plt.subplots()
        ax.hist(ham_nums,weights=ham_weight,color='r',alpha=0.5,label='Ham')
        ax.hist(spam_nums,weights=spam_weight,color='y',alpha=0.5,label='Spam')
        ax.set(title='Freq. of Numbers')
        plt.legend()
        plt.show()
