import numpy as np
import pandas as pd

df = pd.DataFrame(np.random.randint(0,100,(8,5)),columns=list('ABCDE'))

#describe()函数可以查看数据集的统计信息
# print(df.describe([0.25,0.88]).T)

# #head()函数可以查看数据集的前n行数据
# print(df.head(5))

# #tail()函数可以查看数据集的后n行数据
# print(df.tail(5))

# #info()函数可以查看数据集的详细信息
# print(df.info())

# #sample()函数可以随机抽取数据集的n行数据
# print(df.sample(5))


##空值处理:pandas会自动把none值处理成np.nan,np.nan为float类型,而python的none是object类型,float计算效率更高

#isnull()函数可以查看数据集中哪些数据是空值 
#notnull()函数可以查看数据集中哪些数据不是空值
df.loc[1,'A'] = np.nan
df.loc[2,'B'] = np.nan
df.loc[3,'B'] = np.nan
#查看某列空值占比
print(df.isnull().mean())

#空值填充
# df.fillna(value=0,inplace=True)
print(df)
# 用每列的均值填充空值
# df1 = df.fillna(value=df.mean())
# print(df1)
#使用临近值填充,limit限制填充次数
df2 = df.ffill(axis=1,limit=1)
print(df2)
df3 = df2.bfill(axis=0)
print(df3)
#删除空值
df4 = df.dropna(axis=0,how='any',subset=['A','B'])
print(df4)