#knn：分类模型
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split

#加载数据集
iris = datasets.load_iris()

#划分训练集和测试集
X,y = iris.data,iris.target
train_x,test_x,train_y,test_y  = train_test_split(X,y,test_size=0.25,random_state=666)
print("真实标签",test_y)

#训练模型
# n_neighbors:最近样本数量，默认为5
model = neighbors.KNeighborsClassifier(n_neighbors=3)
model.fit(train_x,train_y)

#预测
prediction = model.predict(test_x)
print("预测标签",prediction)

#评估
#print("准确率：",(test_y == prediction).sum()/len(test_y))
print("准确率：",model.score(test_x,test_y))