import numpy as np
import pandas as pd

a1 = np.random.randint(0, 100, (3,4))

level1 = ['上','下']
level2 = ['math','chinese']

columns = pd.MultiIndex.from_product([level1, level2], names=['学期','科目'])

df = pd.DataFrame(data=a1, columns=columns)

print(df)

# print(df.loc[0,('上','math')])
# print(df.iloc[:,:2])

#xs方法:根据标签选择数据
print(df.xs('chinese',level='科目',axis=1).loc[:,'上'])