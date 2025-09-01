#逻辑回归：分类模型(标签是类别)
# 数据集内容：
# 数据 (data)：一个形状为 (150, 4) 的数组，包含150个样本，每个样本有4个特征：
# 花萼长度 (sepal length) (cm)
# 花萼宽度 (sepal width) (cm)
# 花瓣长度 (petal length) (cm)
# 花瓣宽度 (petal width) (cm)
# 目标 (target)：一个形状为 (150,) 的数组，包含每个样本对应的类别标签（0, 1, 2），分别代表：
# 0: Iris Setosa (山鸢尾)
# 1: Iris Versicolour (变色鸢尾)
# 2: Iris Virginica (维吉尼亚鸢尾)

from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split #划分训练集和测试集

#植物分类的数据集
iris = datasets.load_iris()  #dict  data,target

#data:特征值  target:标签值（0,1,2）
X,y = iris.data,iris.target

model = linear_model.LogisticRegression(max_iter=1000) #创建模型
model.fit(X,y) #训练模型
print(model)
#预测
print(model.predict([[5.1,3.5,1.4,0.2]]))

#划分训练集和测试集
train_x,test_x,train_y,test_y  = train_test_split(X,y,test_size=0.25)
#真实标签
print("真实标签：",test_y)

#训练模型
model.fit(train_x,train_y)
#预测
prediction = model.predict(test_x)
print("逻辑回归预测标签：",prediction)

#准确率
#print((test_y == prediction).sum()/len(test_y))
print("准确率：",model.score(test_x,test_y))