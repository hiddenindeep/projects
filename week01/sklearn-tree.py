#决策树：分类模型
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split

#加载数据集
iris = datasets.load_iris()

#划分训练集和测试集
X,y = iris.data,iris.target
train_x,test_x,train_y,test_y  = train_test_split(X,y,test_size=1)
print("真实标签",test_y)

#训练模型
model = tree.DecisionTreeClassifier()
model.fit(train_x,train_y)

#预测
prediction = model.predict(test_x)
print("预测标签",prediction)

#评估
#print("准确率：",(test_y == prediction).sum()/len(test_y))
print("准确率：",model.score(test_x,test_y))

#模型的最终表现有随机性，随机性来源：数据集划分、模型初始化参数