from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

import pandas as PANDA


data_frame = PANDA.read_csv('cleaned_data.csv')


feature_column_names = data_frame.columns[0:-1]

predicted_class_name = data_frame.columns[-1]

x = data_frame[feature_column_names].values
y = data_frame[predicted_class_name].values

split_test_size = 30

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split_test_size, random_state=42)


print("{0:0.2f}% in training set".format((len(x_train)/len(data_frame.index)) * 100))
print("{0:0.2f}% in test set".format((len(x_test)/len(data_frame.index)) * 100))


print("# rows in dataframe {0}".format(len(data_frame)))
print("# rows missing glucose_conc: {0}".format(len(data_frame.loc[data_frame['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(data_frame.loc[data_frame['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(data_frame.loc[data_frame['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(data_frame.loc[data_frame['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(data_frame.loc[data_frame['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(data_frame.loc[data_frame['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(data_frame.loc[data_frame['age'] == 0])))


fill_zero = Imputer(missing_values=0, strategy='mean', axis=0)

x_train = fill_zero.fit_transform(x_train)
y_train = fill_zero.fit_transform(y_train)

train_model = GaussianNB()
train_model.fit(x_train, y_train.ravel())


prediction_from_test_data = train_model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction_from_test_data)
print("Accuracy of our naive bayes model is: {0:0.4f}".format(accuracy))

