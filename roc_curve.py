import numpy as np
import matplotlib.pyplot as plt
from model import NaiveBayesModel, RandomForestClassifierModel, LogisticRegressionModel

nb = NaiveBayesModel('cleaned_data.csv')
naive_bayes = np.array(nb.get_confusion_metrics()[0]/100)

rf = RandomForestClassifierModel('cleaned_data.csv')
random_forest = np.array(rf.get_confusion_metrics()[0]/100)

lr = LogisticRegressionModel('cleaned_data.csv')
logistic = np.array(lr.get_confusion_metrics()[0]/100)

plt.scatter(naive_bayes[0], naive_bayes[1], label = 'Naive Bayes', facecolors='black', edgecolors='orange', s=300)
plt.scatter(random_forest[0], random_forest[1], label = 'Random Forest', facecolors='blue', edgecolors='black', s=300)
plt.scatter(logistic[0], logistic[1], label = 'Logistic Regression', facecolors='orange', edgecolors='orange', s=300)

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc='lower center')

plt.show()
