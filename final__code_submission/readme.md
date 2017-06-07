run.py will generate all graphs and outputs.

Close each figure window to see the next one.

Because Naive Bayes takes so long to run, a .pickle with a pretrained classifier is included to reduce the amount of time it takes to run the code. However, classifying still will take a long time, so please be patient. If you want to see the Naive Bayes Confusion matrix, you can run confusion_matrix_nb.py on the precalculated dataset (nb_output_precalculated.csv). To do that, change the path in confusion_matrix_nb.py to "outputs/nb_output_precalculated.csv"
If you are patient and get an output from naive_bayes_pre_trained.py, you can create a confusion matrix on that output by changing the path to "outputs/nb_output.csv"
naive_bayes_w_training.py will run train a new classifier for Naive Bayes
