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

