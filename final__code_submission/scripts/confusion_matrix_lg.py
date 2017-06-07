# Confusion Matrix: Logistic Regression

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def plot_confusion_matrix(df_confusion, title='Logistic Regression Confusion Matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()

def main():
    # Load results from classifiers to process:

    inpf1 = pd.read_csv("scripts/outputs/log_reg_output.csv")
    
    actual = inpf1['Actual'].tolist()
    predictions = inpf1['Predicted'].tolist()

    y_actu = pd.Series(actual, name='Actual')
    y_pred = pd.Series(predictions, name='Predicted')
    df_confusion = pd.crosstab(y_actu, y_pred)
    
    plot_confusion_matrix(df_confusion)

if __name__ == '__main__':
    main()
