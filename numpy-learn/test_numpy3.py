import numpy as np

#数组添加元素
#1.插入的维度要保证所有数组的长度是相同的
#2.如果维度不同，数组会被扁平处理
a = np.array([[1,2,3,4,5],[1,2,3,4,5],[1,2,3,4,5]])
print(a)
b = np.append(a, [[6,7,8,9,10]], axis=0)
# print(b)
c = np.append(a, np.ones((3,1)), axis=1)
# print(c)

#插入元素
#如果未提供轴，则数组会被展开
d = np.insert(a,2,[6,7,8,9,10],axis=0)
# print(d)
e = np.insert(a,2,[6,7,8,9,10])
# print(e)

#删除元素
#如果未提供轴，则数组会被展开
f = np.delete(a,2,axis=0)
# print(f)

#数组变形
g = np.reshape(a,(5,3))
print(g)