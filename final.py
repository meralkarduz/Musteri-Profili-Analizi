#Konu: Herhangi Bir Kategoride Müşteri Profilinin Belirlenmesi


#Gerekli kütüphaneler projeye import edildi
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

df= pd.read_csv('customer.csv', index_col = 0) #csv dosyası okundu
df.head() #ilk 5 satır okundu
df.rename(columns= {'Genre': 'Gender'}, inplace = True) #"Genre" adlı sütun "Gender" adına çevrildi
df.head() #isim değişiminin kontrolü için ilk 5 satır okundu
df.dtypes #değişkenlerin veri türleri döndürüldü
df.shape #veri kümesinin toplam satır ve sütun sayısı döndürüldü
df.describe()
df.isnull().sum()
df.duplicated()

#yıllık gelir ile harcama puanı arasında ilişki kuruldu
sns.set_style('dark')
sns.scatterplot(x = 'Annual Income (k$)', y = 'Spending Score (1-100)', data = df)
plt.xlabel('Yıllık Gelir (Bin$)')
plt.ylabel('Harcama Puanı (1-100)')
plt.title('Yıllık Gelir ve Harcama Puanı Dağılımı')
X = df.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values

#MinMaxScaler nesnesi oluşturuldu
scaler = MinMaxScaler().fit(X)
print(scaler)
MinMaxScaler()

#yıllık gelir ve harcama puanı 0 ile 1 arasına çekildi
scaler.feature_range
(0, 1)
scaler.transform(X)
array= [[0.        , 0.3877551 ],
       [0.        , 0.81632653],
       [0.00819672, 0.05102041],
       [0.00819672, 0.7755102 ],
       [0.01639344, 0.39795918],]

wcss = []

for i in range(1,11):
    kmeans= KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X) #verilerin k-means'ı bulundu
    wcss.append(kmeans.inertia_) ##her küme için  WCSS değerleri verildi

plt.figure(figsize = (12,6))
plt.grid()
plt.plot(range(1,11),wcss, color='blue', linestyle='dashdot', linewidth = 3,
         marker='o', markerfacecolor='red', markersize=12)
plt.title('Dirsek Noktası Grafiği')
plt.xlabel('Küme Sayısı')
plt.ylabel('WCSS')
plt.show

kmeans= KMeans(n_clusters = 5, init = 'k-means++') #sınıf nesnesi başlatılır
label= kmeans.fit_predict(X) #veri noktalarının her biri için bir küme numarası döndürüldü
print(label)

print(kmeans.cluster_centers_)

plt.figure(figsize=(8,8))
plt.scatter(X[label == 0,0], X[label== 0,1], s=50, c='blue', label='1. Küme') #1.küme çizildi
plt.scatter(X[label == 1,0], X[label== 1,1], s=50, c='red', label='2. Küme') #2.küme çizildi
plt.scatter(X[label == 2,0], X[label== 2,1], s=50, c='green', label='3. Küme') #3.küme çizildi
plt.scatter(X[label == 3,0], X[label== 3,1], s=50, c='yellow', label='4. Küme') #4.küme çizildi
plt.scatter(X[label == 4,0], X[label== 4,1], s=50, c='orange', label='5. Küme') #5.küme çizildi
plt.scatter(kmeans.cluster_centers_ [:,0], kmeans.cluster_centers_ [:,1], s= 100, c='black', marker= '+', label='Merkez') #merkez noktası çizildi
plt.title('Müşteri Grubu')
plt.xlabel('Yıllık Gelir (Bin$)')
plt.ylabel('Harcama Puanı (1-100)')
plt.legend()
plt.show()