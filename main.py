import numpy as np
import pandas as pd

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

g = pd.Series([10,12,14,15],index=['a','b','c','d'])
print(g)
#
data = {'Kraj':['Belgia','Indie','Brazylia'],
        'Stolica':['Bruksela','New Dheli','Brasilia'],
        'Populacja':[1023893,3948383,5838923]
        }
df = pd.DataFrame(data)
print(df)

# daty = pd.date_range('20220507',periods=5)
# print(daty)
#
# df2 = pd.DataFrame(np.random.rand(5,4),index=daty,columns=list('ABCD'))
# print(df2)
#
# df3 = pd.read_csv('dane.csv',header=0,sep=';',decimal='.')
# print(df3)
#
# xlsx = pd.ExcelFile('datasets/imiona.xlsx')
# df4 = pd.read_excel(xlsx,header=0)
# print(df4)
# print(df4.head(10))
# print(df4.tail(10))
#
# df3.to_csv('dane2.csv',index=False)
# df4.to_excel('imiona2.xlsx',sheet_name='dane')

print(g['a'])
print(g.a)
print(df[0:1])
print(df['Kraj'])
print(df.Kraj)
print(df.iloc[[0],[0]])
print(df.loc[[0],['Kraj']])
print(df.at[0,'Kraj'])