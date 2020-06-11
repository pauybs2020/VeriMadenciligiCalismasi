#Kullanılan Kütüphaneler
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

#Veri setini Python’ a Dâhil Etme
dataset = pd.read_csv('C:/Users/55asl/OneDrive/Masaüstü/three_cancer.csv')
#Bağımlı ve bağımsız değişkenlerin oluşturulması
X = dataset.values[:, 0:13]
Y = dataset.values[:, 13]
#Eğitim ve Test Verilerine Oluşturma
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=20, random_state=70)
#Veri Setindeki Kanserleri Tanıya Göre Gruplama
print("Lojistik Regresyon Sınıflandırma Algoritması")
print(dataset.groupby('TANI').size())

#Logisctic Regression Sınıflandırma Algoritması
# lr = LogisticRegression()
# lr.fit(X_train,Y_train)
# predictions = lr.predict(X_test)


#Support Vector Machine(SVM) Sınıflandırma Algoritması
# svc = SVC(kernel = "linear",random_state=70)
# svc.fit(X_train, Y_train)
# predictions = svc.predict(X_test)

# #K komşu Sınıflandırma Algoritması
# k = KNeighborsClassifier(n_neighbors =4,weights='uniform', p=1)
# k.fit(X_train,Y_train)
# predictions = k.predict(X_test)

# #Linear Discriminant Analysis Sınıflandırma Algoritması
# lda =LinearDiscriminantAnalysis()
# lda.fit(X_train,Y_train)
# predictions = lda.predict(X_test)

# #Karar Ağaçları Sınıflandırma Algoritması 
# dt=DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=5,random_state=70)
# dt.fit(X_train,Y_train)
# predictions = dt.predict(X_test)

# #Naive Bayes Sınıflandırma Algoritması
# g = GaussianNB(priors=None,var_smoothing=1e-9)
# g.fit(X_train,Y_train)
# predictions = g.predict(X_test)

# Random Forest ensemble algoritması
# rf = RandomForestClassifier(n_estimators=5,criterion="gini",random_state=70)
# rf.fit(X_train, Y_train)
# predictions = rf.predict(X_test)

#Karmaşıklık Matrisi
c=confusion_matrix(Y_test, predictions)
#Hassasiyet ve Özgüllük Değerleri Hesaplama
sensitivity1 = c[0,0]/(c[0,0]+c[0,1])
specificity1 = c[1,1]/(c[1,0]+c[1,1])
#Hesaplamaları Ekrana Bastırma
print('--> Confusion(Karışıklık) Matrisi Değeri\n',c)
print('-----------------------------------------')
print('--> Accuracy(Doğruluk) Degeri ..:',accuracy_score(Y_test, predictions))
print('--> F1 Hesaplama..:',f1_score(Y_test,predictions,average='micro'))
print('--> Sensitivity : ', sensitivity1 )
print('--> Specificity : ', specificity1)
print('--> Presicion Score..:',precision_score(Y_test, predictions, average = 'macro'))
print('--> Recall Score..:',recall_score(Y_test, predictions, average = 'macro'))
print("---------------------")
print('--> Sınıflandırma Raporu\n',classification_report(Y_test, predictions))
print("---------------------")
#Gerçek Değerler ile Tahmin Değelerini Ekrana Bastırma
print("TEST VERİSİ"," ","TAHMİNLER")
for i in range(len(predictions)):
    print('\t',Y_test[i],'\t\t', predictions[i])