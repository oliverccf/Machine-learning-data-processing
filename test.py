from model import NaiveBayesModel, RandomForestClassifierModel

nb = NaiveBayesModel('cleaned_data.csv')

nb.get_predict_accuracy_on_test_data()
nb.get_predict_accuracy_on_train_data()


rf = RandomForestClassifierModel('cleaned_data.csv')

rf.get_predict_accuracy_on_test_data()
rf.get_predict_accuracy_on_train_data()
