import numpy as np

#数组的基本运算
a = np.random.random(size=(3, 4))
b = np.random.randint(1, 10, size=(3, 4))
# print(a)
# print(b)

#矩阵加法:对应位置相加
c = a + b
# print(c)

#矩阵减法:对应位置相减
c = a - b
# print(c)

#矩阵乘法:矩阵乘法
c = a * b
# print(c)

#矩阵除法:矩阵除法
c = a / b
# print(c)

#广播机制：当两个数组的形状不相同时，广播机制会自动将它们扩展到相同的形状，以便进行运算。广播机制遵循以下规则：
#如果两个数组的后缘维度（从后往前数）必须相等，或者其中一个数组为1。
#则认为它们是广播兼容的。广播会在缺失或长度为1的维度上进行。
#eg1:后缘维度相同
a = np.ones((4,3,2))
b = np.random.randint(1, 10, size=(3,2))
# print(a)
# print(b)
c = a * b
# print(c)
#eg2:其中一个数组为1
a = np.arange(3).reshape((3,1))
# print(a)
b = np.arange(3)
# print(b)
c = a + b
# print(c)

#ndarray可以和任意的整数进行广播运算
# print(a + 1)

#聚合函数
data = np.random.randint(1, 10, size=(3, 4))
print(data)
result = data.sum(axis=0) #按列求和
print(result)
result1 = data.sum()#所有元素求和
print(result1)

#np.nan:表示缺失值 与任何数值运算结果都是nan
data = np.random.randint(1, 10, size=10)
data = data.astype(np.float32)
data[2] = np.nan
print(data)
print(data.sum())
#nansum:忽略nan值
result = np.nansum(data)
print(result)

#常用聚合函数：np.mean(),np.std(),np.var(),np.min(),np.max(),np.argmin(),np.argmax()

data = np.random.randint(0, 2, size=10).astype(np.bool_)
print(np.any(data)) #只要有一个元素为True，返回True
np.all(data) #所有元素都为True，返回True