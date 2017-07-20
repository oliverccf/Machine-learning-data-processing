import pandas as PANDA

def pearsons_correlation(a, b):
    return sum([a[i]*b[i] for i in range(len(a))])/Math.sqrt(sum([x*x for x in a])*sum([y*y for y in b]))

data_frame = PANDA.read_csv('pima-data.csv')

if data_frame.isnull().values.any():
    print('data isnt consistance....')
else:
    print(data_frame.head(3))
