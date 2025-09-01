import numpy as np
import pandas as pd

#pandasIO操作主要读取有特定格式的文件，如csv、excel、json、html、sql等
#pandas读取文件时，默认以行为单位，将每一行数据作为一行，以列名作为列名，以列名和行索引作为索引

df = pd.read_excel('./pandas-learn/test_io.xlsx',sheet_name='Sheet1',names=['姓名','年龄','身高','体重'],header=None)
print(df)

df2 = pd.read_csv('./pandas-learn/test_io.csv',names=['姓名','年龄','身高','体重'],header=None,sep=',')
print(df2)

df3 = pd.read_table('./pandas-learn/test_io.csv',names=['姓名','年龄','身高','体重'],header=None,sep=',',encoding='utf-8')
print(df3)

df3.to_excel('./pandas-learn/out.xlsx')