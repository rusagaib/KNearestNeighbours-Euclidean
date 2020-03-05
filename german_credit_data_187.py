import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#membaca file csv
df = pd.read_csv("german_credit_data.csv", delimiter=",")
df.head()
df.info()
# mencetak kumpulan data berdasar colom Purpose
print(df.groupby('Purpose').size())
# selesi data dari tabel
df_baru = df.select_dtypes(include=['object']).copy()
nRows, nCols = df_baru.shape
for myIndex in range(0,nCols):
    headerName = df_baru.columns[myIndex]
    df_baru[headerName] = df_baru[headerName].astype("category")
    df_baru[headerName] = df_baru[headerName].cat.codes
    df[headerName] = df_baru[headerName]
df_numeric = df
df_numeric.head()

#variabel independen
x = df_numeric.drop(["Purpose"], axis = 1)
x.head()

#variabel dependen
y = df_numeric["Purpose"]
y.head()

# unutk menyimpan nilai max sementara dari hasil fungsi kals
eror_rate=[]

# untuk mengubah data dict ke list
def kelompok(*scores):
    list=[x for x in scores]
    return list

# fungsi unutk menghitung nilai max tiap nilai di n_Neighbour dalam dict max_acc
def hitung_max_dict(scores):
    newlist=[]
    for x in scores.values():
        newlist+=kelompok(x)
    key=max(scores, key=scores.get)
    return key, max(newlist)

# fungi untuk mengacak nilai KNeighbors dari 1~100
def kals(xtr,xts,ytr,yts):
    eror_rate=[]
    xtr=x_train
    xts=x_test
    yts=y_test
    ytr=y_train
    for i in range(1,100):
        # Mengaktifkan fungsi klasifikasi
        klasifikasi = KNeighborsClassifier(n_neighbors=i)
        # Memasukkan data training pada fungsi klasifikasi
        klasifikasi.fit(x_train, y_train)
        # Menentukan hasil prediksi dari x_test
        y_pred = klasifikasi.predict(x_test)
        y_pred
        klasifikasi.predict_proba(x_test)
        accuracy= accuracy_score(y_test, y_pred)
        eror_rate.append(accuracy)
    # dibuatkan data dict agar mudah untuk mencari nilai max
    max_acc={}
    for i, item in enumerate(eror_rate, start=1):
        # print(i, item)
        max_acc[str(i)]=item
    print('max=',hitung_max_dict(max_acc))

# Main
# for i in np.arange(0.1, 1.0, 0.1): #<-- tadinya mau pakai ini tapi nilai akurasinya tidak ada yang memuaskan
for i in range(1, 10):
    for j in range(1, 11):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = i, random_state = j)
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_test = scaler.transform(x_test)
        print('ts: ',i,'\nrs: ',j)
        kals(x_train,x_test,y_train,y_test)
        print()
