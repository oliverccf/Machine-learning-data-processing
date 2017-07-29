from sklearn.model_selection import train_test_split
import pandas as PANDA


data_frame = PANDA.read_csv('cleaned_data.csv')

feature_column_names = [
    'num_preg',
    'glucose_conc',
    'diastolic_bp',
    'thickness',
    'insulin',
    'bmi',
    'diab_pred',
    'age'
]

predicted_class_name = ['diabetes']

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

