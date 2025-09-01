import numpy as np
import pandas as pd

score1 = np.random.randint(30, 101, (4,3))
score2 = np.random.randint(30, 101, (4,3))

#期中成绩
df_qz = pd.DataFrame(data=score1,index=['张三','李四','王五','赵六'],columns=['Python','Java','C'])
#期末成绩
df_qm = pd.DataFrame(data=score2,index=['张三','李四','王五','赵六'],columns=['Python','Java','C'])

df_sum = df_qz.add(df_qm, fill_value=0)

print(df_qz)
print(df_qm)
#print(df_sum)

#1.各科平均值
print('期中平均值：\n',df_qz.mean(axis=0))
print('\n')
print('期末平均值：\n',df_qm.mean(axis=0))
print('\n')

#2.张三期中考试java作弊记0分
df_qz.loc['张三','Java'] = 0
print('张三成绩调整后：\n',df_qz)
print('\n')

# 3.期中李四的成绩全部加10
df_qz.loc['李四',:] += 10
print('李四成绩调整后：\n',df_qz)
print('\n')

# 4.期末成绩全部加10
df_qm += 10
print('期末成绩调整后：\n',df_qm)

#5.哪些同学python比java好
print(df_qm.query('Java > Python').index)