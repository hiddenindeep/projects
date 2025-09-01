import numpy as np
import pandas as pd

# 创建一个Series
# 1.用ndarrary构造的Series是一个引用对象，修改Series会同时影响ndarray
arr1 = np.random.randint(0, 10, 5)
s1 = pd.Series(arr1,name='s1')
# print(s1)
# s1[0] = 100
# print(s1)
# print(arr1)

# 2.用列表构造的Series是一个copy对象，修改Series不会影响列表
s2 = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
# print(s2)

# 3.用字典构造的Series，字典的key作为index，字典的value作为value
dict1 = {'name': 'zhangsan', 'age': 18, 'address': 'shanghai'}
s3 = pd.Series(dict1)
# print(s3)
s4 = pd.Series(data=dict1, index=['a', 'age', 'c', 'name'])
# print(s4)

# 4.用标量构造的Series，Series的index是默认的整数序列
s5 = pd.Series(10, index=['a', 'b', 'c', 'd', 'e'])
# print(s5)
# print(s5.a)
#Series的属性：index、values、dtype、shape、ndim、size、empty、name


# Series的运算原则 索引对齐原则
s1 = pd.Series(data=[5,4,1],index=['a','b','c'])
s2 = pd.Series(data=[1,2,3],index=['a','c','d'])
print(s1+s2)
# a    6.0
# b    NaN
# c    3.0
# d    NaN