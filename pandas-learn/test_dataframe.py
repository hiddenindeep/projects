import numpy as np
import pandas as pd

# DataFrame的结构:行索引、列索引、数据
# 与Series的关系:DataFrame是Series的容器

# 使用ndarry创建dataframe
df1 = pd.DataFrame(data=[[1,2,3],[4,5,6],[7,8,9]],index=['a','b','c'],columns=['one','two','three'])
print(df1)

# 使用字典创建DataFrame
name = ['张三','李四','刘亦菲','王五']
score_yuwen = np.random.randint(0, 100, 4)
score_math = np.random.randint(0, 100, 4)
dict1 = {'name': name, 'score_yuwen': score_yuwen, 'score_math': score_math}
df2 = pd.DataFrame(data=dict1)
# print(df2)

#DataFrame的属性:shape、index、columns、values
# print(df2.shape)
# print(df2.index)
# print(df2.columns)
# print(df2.values)
# print(df2.dtypes)

# DataFrame和Series运算默认就是列索引对齐原则
s1 = pd.Series(data=[1,2,3],index=['one','two','three'])
# print(s1)
df2 = pd.DataFrame.add(df1, s1, axis='columns')
# print(df2)
s3 = pd.Series(data=[1,2,3],index=['d','b','c'])
df3 = pd.DataFrame.add(df1, s3, axis='index')
# print(df3)

df4 = pd.DataFrame(data=s3, index=['d','b','c'], columns=['one'])
print(df4)
print(pd.DataFrame.add(df1, df4, axis='index', fill_value=0))

#loc和iloc

#where,mask,filter,query

#单层索引
df4.index = pd.Index(['A','B','C'], name='name')
print(df4)


