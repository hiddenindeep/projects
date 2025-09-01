import numpy as np
import pandas as pd

index = pd.Index(['lucy','tom','jack'], name='姓名')
columns = pd.MultiIndex.from_product([['期中','期末'],['python','java','c']], names=['学期','科目'])
data = np.random.randint(0, 100, (3,6))
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df)

#获取lucy期中最高分的学科
lucy_score = df.loc['lucy','期中']
print(f"lucy期中最高分的学科：{lucy_score.idxmax()},分数：{lucy_score.max()}") # type: ignore

#计算tom期中期末各学科的平均成绩
tom_all = df.loc['tom'].unstack()
print(tom_all.mean())

#jack期中python+20
df.loc['jack',('期中','python')] += 20 # type: ignore
print(df)
