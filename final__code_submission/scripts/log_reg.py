# ## CSCI 183 Data Science
# ### Spam Filtering for Short Messages: Logistic Regression
# #### Ryan Johnson, Grace Nguyen, and Raya Young

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


def main():
    data = pd.read_csv("../data/spamcollectiondata.tsv", sep='\t', names = ["Category", "Message"])

    print("=========================== Logistic Regression ===========================")

    message_data = [word.lower() for word in data['Message']]
    category = data['Category'].tolist()
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

    train_df = pd.DataFrame(training_set)
    train_df.columns = ['Lists','Labels']
    train_df.head()
    y = np.array(train_df.Labels)

    for i in range(len(y)):
        if y[i] == 'ham':
            y[i] = 1
        else:
            y[i] = 0

    y = np.ravel(y)


    def list_to_dict(words_list):
        return dict([(word, True) for word in words_list])

    training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in training_set]

    def generate_words_vector(training_set):
        words_vector = []
        for review in training_set:
            for word in review[0]:
                if word not in words_vector: words_vector.append(word)
        return words_vector

    def generate_X_matrix(training_set, words_vector):
        no_reviews = len(training_set)
        no_words = len(words_vector)
        X = np.zeros(shape=(no_reviews, no_words+1))
        for ii in range(0,no_reviews):
            X[ii][0] = 1
            review_text = training_set[ii][0]
            total_words_in_review = len(review_text)
            rt = list(review_text)
            for word in rt:
                word_occurences = rt.count(word)
                word_index = words_vector.index(word)+1
                X[ii][word_index] = word_occurences / float(total_words_in_review)
        return X

    words_vector = generate_words_vector(training_set_formatted)
    X = np.array(generate_X_matrix(training_set_formatted, words_vector))


    print("---------------------------------------------------------------------")
    print("Dataset Analysis:")
    # What percentage is ham?
    print("...Percentage ham: " + str(y.mean()))


    X_train, X_test, y_train, y_test = train_test_split(X, train_df.Labels, test_size=0.5, random_state=1)
    model = LogisticRegression()
    model.fit(X_train,y_train)

    probs = model.predict_proba(X_test)

    predicted = model.predict(X_test)

    act = 1 - y[:len(predicted)]
    pred = []

    for result in predicted:
        if result == 'spam':
            pred.append(1)
        else:
            pred.append(0)

    # 1 is spam, 0 is ham
    compare = pd.DataFrame({'Actual':act,'Predicted':pred,
                       'Prob is Ham':probs[:,0],
                       'Prob is Spam':probs[:,1]})

    print("\n---------------------------------------------------------------------")
    print("Comparison of spam to ham: ")
    print(compare[compare['Prob is Spam']>=0.5].head(),'\n')

    print("\n---------------------------------------------------------------------")
    print("Results of classification:")
    print(compare)

    compare.to_csv('scripts/outputs/log_reg_output.csv',sep=',')

    print("\n---------------------------------------------------------------------")
    print("Confusion Matrix:")
    print(metrics.confusion_matrix(y_test,predicted))

    print("\n---------------------------------------------------------------------")
    print("Classification Report:")
    print(metrics.classification_report(y_test,predicted))

    print("\n---------------------------------------------------------------------")
    print("Classification Summary:")
    print(pd.DataFrame(y_test).describe())


    # Comparing Ratio of Spam/Ham to Model's Ratio of
    print("\n---------------------------------------------------------------------")
    print("Actual Ratio of Spam/Ham: " + str(1451.0/1672))
    print("Model's Ratio: " + str(1669.0/1672))

    print("\n---------------------------------------------------------------------")
    print("Accuracy Score:")
    print(metrics.accuracy_score(y_test,predicted))

    print("\n********** Full results for logistic regression can be seen in log_reg_output.csv *****************")

    # ### Testing Our Model
    print("\n--------------------TESTING OUR MODEL--------------------------------")

    test_phrase_1 = "lt gt go come ok love"
    test_phrase_2 = "call free mobil co.uk"


    arr_1 = np.zeros(len(X[0]))
    arr_2 = np.zeros(len(X[0]))

    for ii in range(len(test_phrase_1.split())):
        review_text = test_phrase_1.split()
        total_words_in_review = len(review_text)
        rt = list(review_text)
        for word in rt:
            word_occurences = rt.count(word)
            if word in words_vector:
                word_index = words_vector.index(word)+1
                arr_1[word_index] = word_occurences / float(total_words_in_review)

    for ii in range(len(test_phrase_2.split())):
        review_text = test_phrase_2.split()
        total_words_in_review = len(review_text)
        rt = list(review_text)
        for word in rt:
            word_occurences = rt.count(word)
            if word in words_vector:
                word_index = words_vector.index(word)+1
                arr_2[word_index] = word_occurences / float(total_words_in_review)

    arr_1[0] = 1
    arr_2[0] = 1

    pct_1 = model.predict_proba(arr_1.reshape(1,-1))[0][0]*100
    pct_2 = model.predict_proba(arr_2.reshape(1,-1))[0][0]*100

    ###########################

    result_1 = result_2 = 'SPAM'

    if(pct_1>50):
        result_1 = 'HAM'
    if(pct_2>50):
        result_2 = 'HAM'

    print("\"{}\" --> {} \n\t{}% Ham / {}% Spam.".format(test_phrase_1,result_1,round(pct_1,1),round(100-pct_1,1)),"\n")
    print("\"{}\" --> {} \n\t{}% Ham / {}% Spam.".format(test_phrase_2,result_2,round(pct_2,1),round(100-pct_2,1)))


if __name__ == '__main__':
    main()
