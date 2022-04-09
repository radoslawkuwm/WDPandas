import numpy as np

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
#

# zad1
a = np.arange(3,46,3)
print(a)

#zad2
b = np.array([1.25,3.74,6.23,67.32])
c = b.astype('int64')
print(c)

#zad3
def zad3(n):
    return np.arange(n*n).reshape([n,n])

print(zad3(5))

#zad4
print(np.logspace(1,6))

