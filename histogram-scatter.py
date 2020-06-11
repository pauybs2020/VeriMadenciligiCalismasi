import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import numpy as np
# Veri setinin yüklenmesi
dataset = pd.read_csv('C:/Users/55asl/OneDrive/Masaüstü/three_cancer.csv')
# print(cancer_dataset)
#Bağımlı ve bağımsız değişkenlerin oluşturulması
X = dataset.values[:, 0:13] # hastaid haric ilk 19 sutun bagimsiz degisken
Y = dataset.values[:, 13]    # 21. sutun Tani yani bagimli degisken

# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 14].values


#Veri kümesinin eğitim ve test verileri olarak ayrılması
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.10, random_state=70)
# print(Y_test)
# print(dataset.groupby('TANI').size())

# histogram
# plt.style.use('ggplot')
# # bins=np.arange(0,200,20)
# # plt.xticks(bins)
# # plt.hist(dataset.B,edgecolor='red',bins=bins)
# plt.hist(dataset)

# scatter_matrix(dataset.loc[:,'A':'G'],figsize = (12,10),diagonal='kde')



# # ## kutu grafigi
# dataset.A.plot(kind='box', subplots=True, sharex=False, sharey=False)
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
# plt.savefig('box.svg', format='svg', dpi=1200)
# plt.show()
# #

