import pandas as PANDA
import matplotlib.pyplot as plt


def pearsons_correlation(a, b):
    return sum([x*y for x,y in zip(a,b)])/Math.sqrt(sum([x*x for x in a])*sum([y*y for y in b]))


def corr_heatmap(data_frame, size=11):
    correlation = data_frame.corr()
    fig, heatmap = plt.subplots(figsize=(size, size))
    heatmap.matshow(correlation)
    plt.xticks(range(len(correlation.columns)), correlation.columns)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.show("heat map")


if __name__ == "__main__":

    data_frame = PANDA.read_csv('pima-data.csv')

    if data_frame.isnull().values.any():
        print("data isn't consistence....")
    else:
        corr_heatmap(data_frame)
        print("Enter column name to delete: ")
        columns = input().split()
        for column in columns:
            del data_frame[column]
            data_frame.head()

        print('After Cleaning......')
        corr_heatmap(data_frame)

        print(data_frame.head())

        map_diabetes = {True: 1, False: 0}
        data_frame['diabetes'] = data_frame['diabetes'].map(map_diabetes)

        data_frame.to_csv('cleaned_data.csv', index=False, encoding='utf-8')

