import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# a = np.array([1, 2, 3],dtype='float64')
# print(a)
# a = np.arange(1,5,0.5)
# print(a)
# print(a.dtype)
# print(type(a))
# print(a.shape)
# print(a.ndim)
#
# b = np.array([np.arange(1,6,1),np.arange(2,12,2)])
# print(b)
# print(b.shape)
# print(b.ndim)
#
# zera = np.zeros((5,5),dtype='int32')
# print(zera)
#
# jedynki = np.ones((5,5))
# print(jedynki)
#
# puste = np.empty((2,2))
# puste[1][1] = 5
# print(puste)
#
# a= np.linspace(1,2,5,endpoint=False)
# print(a)
#
# b,c = np.indices((5,5))
# print(b)
# print(c)
#
# d, e = np.mgrid[0:4, 0:4]
# print(d)
# print(e)
#
# f = np.diag([x for x in range(5)])
# print(f)
#
# g = np.fromiter(range(7),dtype='int32')
# print(g)
#
# marcin = 'Marcin'
# # mar = np.frombuffer(marcin,dtype='S6')
# # print(mar)
# mar_1 = np.array(list(marcin))
# print(mar_1)
# mar_3 = np.fromiter(marcin, dtype='S1')
# mar_4 = np.fromiter(marcin, dtype='U1')
# print(mar_3)
# print(mar_4)

# a = np.ones((2,2))
# b = np.ones((2,2))
# c = a + b
# print(c)
#
# a = np.arange(10)
# print(a)
#
# s = slice(2,7,2)
# print(a[s])
#
# print(a[1:])
# print(a[1:6:2])

# a = np.arange(25)
# mat = a.reshape((5,5))
# print(mat)
# print(mat[1:,1:])
# print(mat[:,1:2])
# print(mat[2:5,1:3])

# a = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
# # print(a)
# rows = np.array([[0,0],[3,3]])
# cols = np.array([[0,2],[0,2]])
# b = a[rows,cols]
# print(b)
# #
#
# # zad1
# a = np.arange(3,46,3)
# print(a)
#
# #zad2
# b = np.array([1.25,3.74,6.23,67.32])
# c = b.astype('int64')
# print(c)
#
# #zad3
# def zad3(n):
#     return np.arange(n*n).reshape([n,n])
#
# print(zad3(5))
#
# #zad4
# print(np.logspace(1,6))
#
#

# a = np.array([20,30,40,50])
# b = np.arange(4)
# c=a+b
#
# print(c)
# d = np.sqrt(b)
#
# print(d)
#
# e = d + c;
#
# print(e)

# a = np.arange(16).reshape(4,4)
# print(a.sum())
# print(a.sum(axis=0))
# print(a.sum(axis=1))
# print(a.cumsum(axis=1))

# a = np.arange(3)
# b = np.arange(3)
# c = np.dot(a,b)
# print(c)
# print(a.dot(b))
#
# d = np.arange(3)
# e = np.array([[0],[1],[2]])
# print(d.dot(e))
#
# f = np.array([[2,4,5],[5,1,7]])
# g = np.array([[2,3],[4,2],[6,1]])
# print(np.dot(f,g))

# a = np.arange(6).reshape((3,2))
# print(a)
#
# for b in a:
#     for c in b:
#         print(c)
#
#
# a = np.arange(12).reshape((3,4))
# print(a)
#
# b = a.reshape((4,3))
# print(b)
#
# c = b.ravel()
# print(c)
#
# print(b.T)
#
#
# s = pd.Series([1,3,5,'a',5.5])
# print(s)

# s = pd.Series([10, 12, 14, 15], index=['a', 'b', 'c', 'd'])
# print(s)
# print(s['a']) #10
# print(s.b,'\n')#12
# print(s[s>12],'\n')
# print(s.where(s>12,'wartosc mniejsza niz zalozenie(12)'))
# seria = s.copy()
# seria.where(seria>12,'warunek nie spelniony',inplace=True)
# print(seria)
# print(s[(s>10) & (s<15)])
# s['e'] = 17
# print(s)



# data = {'Kraj': ['Belgia', 'Indie', 'Brazylia'],
#         'Stolica': ['Bruksela', 'New Dheli', 'Brasilia'],
#         'Populacja': [48675467, 239483833, 58675467]
#         }
# df = pd.DataFrame(data)
# print(df.Kraj,'\n')
# print(df[df['Populacja']>3000000])
# df.loc[3] = 'nowy element'
# df.loc[4] = ['Polska','Warszawa',38675467]
#
# print(df)
#
# df.drop([3],inplace=True)
# print(df)
#
# df['Kontynent'] = ['Europa','Azja','Ameryka Poludniowa','Europa']
# print(df)
#
# df.sort_values('Kraj',inplace=True)
# print(df)

# grupa = df.groupby('Kontynent').agg({'Populacja':'sum'})
# print(grupa.get_group('Europa'))

# grupa.plot(kind='bar',xlabel ='Kontynent',ylabel='Populacja',title='Populacja na kontynent',rot=0)

# wykres = grupa.plot.bar()
# wykres.set_xlabel = 'Kontynenty'
# wykres.set_ylabel = 'Populacja w mld'
# wykres.tick_params(axis='x',labelrotation=0)
# wykres.set_title('Populacja na kontynent')
# plt.show()

# print(df.groupby('Kontynent').agg({'Populacja'})) #fix

#print(df.iloc([[0], [0]]))
# df = pd.DataFrame(data)
# print(df)
#
# daty = pd.date_range('20220507',periods=5)
# print(daty)
#
# df2 = pd.DataFrame(np.random.rand(5,4),index=daty,columns=list('ABCD'))
# print(df2)
# #
# df3 = pd.read_csv('dane.csv',header=0,sep=';',decimal='.') #wczytanie pliku csv
# # print(df3.sample(10,replace=True),'\n') # 10 losowych, replace = True bo ilosc danyc w ramce <10
# #
# xlsx = pd.ExcelFile('datasets/imiona.xlsx') #otworzenie pliku xlsx
# df4 = pd.read_excel(xlsx, header=0) #wczytanie pliku xlsx
# # print(df4)


# print(df4.head(10),'\n') # pierwsze 10
# print(df4.tail(10),'\n') # ostatnie 10

# print(df4.sample(10),'\n') # losowy wiersz

# df3.to_csv('dane2.csv',index=False) #zapis do csv
# df4.to_excel('imiona2.xlsx',sheet_name='dane') #zapis do xlsx
#

# print(df[0:1]) #wypianie wiersza 0 do 1 (bez 1)
# print(df['Kraj']) #wartosci w kolumnie z headerem 'Kraj'
# print(df.Kraj) #jw
# print(df.iloc[[0],[0]])
# print(df.loc[[0],['Kraj']])
# print(df.at[0,'Kraj'])

# grupa = df3.groupby('Imię i nazwisko').agg({'Wartość zamówienia':['sum']})
# print(grupa)
# grupa.plot(kind='pie',subplots=True,autopct='%.2f%%',fontsize=20,colors=['red','green'],figsize=(8,6))
# plt.legend(loc='upper left')
# plt.savefig('plot.png')
# plt.show

# seria = pd.Series(np.random.randn(1000))
# seria = seria.cumsum()
#
# seria.plot()
# plt.show()

# plt.plot([1,2,3,4],[1,4,9,16],'ro:',label='linia')
# plt.ylabel('wartpsco z wektora')
# plt.show()
#
# plt.plot([1,2,3,4], [1,4,9,16],'r:')
# plt.plot([1,2,3,4], [1,4,9,16],'bo')
#
# # plt.axis([0,6,0,20])
# plt.show()

# t = np.arange(0,5,0.1)
#
# plt.plot(t, t, 'r-', t, t**2,'b:',t,t**3,'g^')
# plt.legend(labels=['liniowa','kwadratowa','szescienna'], loc='center')
# plt.show()

# x = np.linspace(0,2,100)
#
# plt.plot(x,x,label='liniowa')
# plt.plot(x,x**2,label = 'kwadrtatowa')
# plt.plot(x,x**3, label= "szescienna")
# plt.xlabel('etykieta x')
# plt.ylabel('etykieta y')
# plt.title('wykres trzech linii')
# plt.savefig('plot.png')
# im1 = Image.open('plot.png')
# im1 = im1.convert('RGB')
# im1.save('plot.jpg')

# x = np.linspace(1,20,100)
# print(x)
# plt.plot(x,1/x,'bo-', label='funkcja')
# plt.ylabel('etykieta y')
# plt.xlabel('etykieta x')
# plt.legend(loc = 'best')
# plt.show()

# x = np.arange(0,10,0.1)
# plt.plot(x,np.sin(x),'ro:',label='sinus')
# plt.xlabel('wartosc x')
# plt.ylabel('sinus')
# plt.show()

#siatka
#
# x1 = np.arange(0,2,0.02)
# x2 = np.arange(0,2,0.02)
#
# y1 = np.sin(2 * np.pi * x1)
# y2 = np.cos(2 * np.pi * x2)
#
# # plt.subplot(2,1,1)
# # plt.plot(x1,y1)
# # plt.ylabel('sinus(x)')
# # plt.title('wykres sin(x)')
# #
# # plt.subplot(2,1,2)
# # plt.plot(x2,y2,'r-')
# # plt.ylabel('cosinus(x)')
# # plt.xlabel('x')
# #
# # plt.subplots_adjust(hspace=0.5)
# # plt.show()
#
# fig, axs = plt.subplots(3,2)
# # print(type(fig))
# # print(type(axs))
#
# axs[0,0].plot(x1,y1, 'g-')
# axs[0,0].set_xlabel('x')
# axs[0,0].set_ylabel('sin(x)')
# axs[0,0].set_title('Wykres sin(x)')
#
# axs[1,1].plot(x2,y2,'r-')
# axs[1,1].set_xlabel('x')
# axs[1,1].set_ylabel('cos(x)')
# axs[1,1].set_title('wykres cos(x)')
#
#
# axs[2,0].plot(x2,y2,'b:')
# axs[2,0].set_xlabel('x')
# axs[2,0].set_ylabel('cos(x)')
# axs[2,0].set_title('wykres cos(x)')
#
# fig.delaxes(axs[0,1])
# fig.delaxes(axs[1,0])
# fig.delaxes(axs[2,1])
#
# plt.subplots_adjust(hspace=0.5,wspace=0.25)
# plt.show()

# data = {
#         'a':np.arange(50),
#         'c':np.random.randint(0,51,50),
#         'd':np.random.randn(50)
#         }
# data['b'] = data['a'] + 10 * np.random.randn(50)
# data['d'] = np.abs(data['d']) * 100
#
# plt.scatter(data=data, x='a',y='b',c='c',cmap='Accent',s='d')
# plt.xlabel('wartosci z klucza a')
# plt.ylabel('wartosci z klucza b')
# plt.show()

# data = {'Kraj': ['Belgia', 'Indie', 'Brazylia'],
#         'Stolica': ['Bruksela', 'New Dheli', 'Brasilia'],
#         'Populacja': [48675467, 1099483833, 258675467]
#         }
# df = pd.DataFrame(data)
# df.loc[3] = ['Polska','Warszawa',38675467]
#
# df['Kontynent'] = ['Europa','Azja','Ameryka Poludniowa','Europa']
#
# grupa = df.groupby('Kontynent')
# etykiety = list(grupa.groups.keys())
# wartosci = list(grupa.agg('Populacja').sum())
#
# print(etykiety)
# print(wartosci)
#
# plt.bar(x=etykiety,height=wartosci, color=['red','green','blue'])
# plt.xlabel('Kontynent')
# plt.ylabel('Populacja na kontynentach')
# plt.show()

x = np.random.randn(10000)
plt.hist(x,bins=50, facecolor='g',alpha=0.75,density=True)
plt.xlabel('wartosci')
plt.ylabel('Prawdopodobienstwo')
plt.title('Histogram')
plt.show()