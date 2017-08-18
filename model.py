import pandas as PANDA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


class BaseModel:
    def __init__(self, file_name="", train_model=None):
        self.file_name = file_name
        self.train_model = train_model
        self.data_frame = PANDA.read_csv(file_name)
        self.feature_column_names = self.data_frame.columns[0:-1].values
        self.predicted_class_name = [self.data_frame.columns[-1]]
        self.x = self.data_frame[self.feature_column_names].values
        self.y = self.data_frame[self.predicted_class_name].values
        split_test_size = int((len(self.data_frame.index) * 30) / 100)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,
                                                                                self.y,
                                                                                test_size=split_test_size,
                                                                                random_state=42)
        fill_zero = Imputer(missing_values=0, strategy='mean', axis=0)
        self.x_train = fill_zero.fit_transform(self.x_train)
        self.x_test = fill_zero.fit_transform(self.x_test)
        self.train_model.fit(self.x_train, self.y_train.ravel())

    def show_data_frame_status(self):
        print("{0:0.2f}% in training set".format((len(self.x_train) / len(self.data_frame.index)) * 100))
        print("{0:0.2f}% in test set".format((len(self.x_test) / len(self.data_frame.index)) * 100))

        print("# rows in dataframe {0}".format(len(self.data_frame)))
        for column in self.feature_column_names:
            print("#rows missing in {0} : {1}".format(column, len(self.data_frame.loc[self.data_frame[column] == 0])))

    def get_predict_accuracy_on_test_data(self):
        prediction_from_test_data = self.train_model.predict(self.x_test)
        accuracy = metrics.accuracy_score(self.y_test, prediction_from_test_data)
        print("Accuracy of our naive bayes model on test data is: {0:0.4f}".format(accuracy))

    def get_predict_accuracy_on_train_data(self):
        prediction_from_train_data = self.train_model.predict(self.x_train)
        accuracy = metrics.accuracy_score(self.y_train, prediction_from_train_data)
        print("Accuracy of our naive bayes model on training data is: {0:0.4f}".format(accuracy))


class NaiveBais(BaseModel):
    def __init__(self, file_name=""):
        naive_bais = GaussianNB()
        super().__init__(file_name=file_name, train_model=naive_bais)
