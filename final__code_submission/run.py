#!/usr/bin/python
from importlib.machinery import SourceFileLoader

data_analysis = SourceFileLoader("data_analysis", "scripts/data_analysis.py").load_module()
log_reg = SourceFileLoader("log_reg", "scripts/log_reg.py").load_module()
naive_bayes_pre_trained = SourceFileLoader("naive_bayes_pre_trained", "scripts/naive_bayes_pre_trained.py").load_module()
confusion_matrix_lg = SourceFileLoader("confusion_matrix_lg", "scripts/confusion_matrix_lg.py").load_module()
confusion_matrix_nb = SourceFileLoader("confusion_matrix_nb", "scripts/confusion_matrix_nb.py").load_module()

#Data Analysis
data_analysis.main("Capitals")
data_analysis.main("Numbers")

#Logistic Regression
log_reg.main()
confusion_matrix_lg.main()

#Naive Bayes
naive_bayes_pre_trained.main()
confusion_matrix_nb.main()

