from model import NaiveBayesModel, RandomForestClassifierModel, LogisticRegressionModel

nb = NaiveBayesModel('cleaned_data.csv')

print("accuracy of Naive Bayes Model: " + str(nb.get_predict_accuracy_on_test_data()))
print(nb.get_confusion_metrics())
print(nb.get_classification_report())
nb.get_predict_accuracy_on_train_data()


rf = RandomForestClassifierModel('cleaned_data.csv')

rf.get_predict_accuracy_on_test_data()
rf.get_predict_accuracy_on_train_data()


lgr = LogisticRegressionModel('cleaned_data.csv')

lgr.get_predict_accuracy_on_test_data()
lgr.get_predict_accuracy_on_train_data()
