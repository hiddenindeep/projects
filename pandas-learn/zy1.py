import pandas as pd
import numpy as np

df1 = pd.DataFrame(data=[[1,'s',3],[4,5,6],[7,8,9]],index=['a','b','c'],columns=['one','two','three'])
df2 = pd.DataFrame(data=[[1,2,3],[4,5,6],[7,8,9]],index=['a','b','c'],columns=['one','two','9'])
print(df1)
print(df2)
# d3 = df1.add(df2, axis='index',fill_value=0)
# print(d3)

# print(d3.dtypes)
# print(d3.loc['a']['one'])

# print(df1.add(df2, axis='columns'))