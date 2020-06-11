
from matplotlib.pyplot import plt
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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB 

#Veri setini Python’ a Dâhil Etme
dataset = pd.read_csv('C:/Users/55asl/OneDrive/Masaüstü/three_cancer.csv')
#Bağımlı ve bağımsız değişkenlerin oluşturulması
X = dataset.values[:, 0:13]
Y = dataset.values[:, 13]
#Eğitim ve Test Verilerine Oluşturma
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.12, random_state=70)
#Veri Setindeki Kanserleri Tanıya Göre Gruplama

# x=yas y=geliş 
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,69)
# size = np.random.randint(50,200,69)

# plt.scatter(dataset.D,dataset.TANI)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta Geliş Sayısı')
# plt.show()

#  # x=cinsiyet y=CA19-9
# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.D,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta ALT Değerleri')
# plt.show()

#  # x=cinsiyet y=ygelişsayısı
# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.B,dataset.M,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Cinsiyetleri')
# plt.ylabel('Hasta Geliş Sayısı')
# plt.show()

#  # x=yas y=CA 19-9 
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.C,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta CA 19-9 Değerleri')
# plt.show()

#  # x=yas y=Alt 
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,69)
# size = np.random.randint(50,200,69)
# plt.scatter(dataset.A,dataset.D,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta ALT Değerleri')
# plt.show()

#  # x=yas y=CRP 
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.E,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta CRP Değerleri')
# plt.show()

#  # x=yas y=CEA
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.F,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta CEA Değerleri')
# plt.show()

#  # x=yas y=AFP
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.G,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta AFP Değerleri')
# plt.show()

#  # x=yas y=cinsiyet
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.B,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta Cinsiyetleri')
# plt.show()

#  # x=YAS y=Tani
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.A,dataset.TANI,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta CA 19-9 Değerleri')
# plt.show()

#  # x=TANİ y=CA19-9
# plt.style.use('seaborn-ticks')
# fig=plt.figure()
# ax=plt.axes()

# color = np.random.randint(0,10,60)
# size = np.random.randint(50,200,60)
# plt.scatter(dataset.TANI,dataset.C,s=size,c=color,edgecolor='black',cmap='viridis',linewidth =1,alpha = 1)
# cbar=plt.colorbar()
# cbar.set_label('Seviyeler')
# plt.title('Veri Üzerindeki Saçılım Grafiği')
# plt.xlabel('Hasta Yaşları')
# plt.ylabel('Hasta CA 19-9 Değerleri')
# plt.show()
