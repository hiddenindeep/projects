import numpy as np

scores = np.random.randint(0, 101, 100)
# print(scores)
jg = (scores >= 60).astype(int)
# print(np.mean(jg))

#随机生成一个一维数组，比较其中是否有至少一个数据大于3倍平均值
a = np.random.randn(100)
# print((a > a.mean() * 3).any())

# 生成300行3列的服从标准正态分布的随机数
b = np.random.randn(300,3)
# print(b)
# # 计算每列的标准差，并乘以3
# print(b.std(axis=0)*3)
# # 判断每列中是否有大于标准差乘以3的数
# print((b > b.std(axis=0)*3).any(axis=0))

#检查两个形状相同的数组数值是完全一致?
a = np.ones(shape=(3, 3))
b = np.ones(shape=(3, 3))
print(np.all(a == b))

#遍历数组
for i in a.flat:
    print(i)