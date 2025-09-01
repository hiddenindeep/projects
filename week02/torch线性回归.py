import torch
import numpy as np
import matplotlib.pyplot as plt

#计算y=a*x+b中的a和b

#模拟数据
x_numpy = np.random.rand(100,1) * 10
y_numpy = x_numpy * 2 + np.random.rand(100,1)
x = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

#初始化参数
# a = torch.randn(1,requires_grad=True,dtype=torch.float)
# b = torch.randn(1,requires_grad=True,dtype=torch.float)
model = torch.nn.Linear(1,1)

#定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
#均方误差
#a*x+b << - >> y'

#优化器基于a b 梯度自动更新参数a,b。先传入初始的参数
#optimizer = torch.optim.SGD([a,b],lr=0.01)   #学习率太小，loss变化很小很慢；学习率太大，loss变得很大，不稳定，直接跳过波段最低点
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

#训练模型
num_epochs = 1000 #迭代次数
for epoch in range(num_epochs):
    #前向传播
    #y_pred = a * x + b
    y_pred = model(x)

    #计算损失
    loss = loss_fn(y_pred,y)

    #反向传播和优化
    optimizer.zero_grad() #清空梯度 torch梯度累加
    loss.backward()       #计算梯度
    optimizer.step()      #更新参数

    if (epoch + 1) % 100 == 0:
        #print(f'{epoch + 1}/{num_epochs} loss:{loss.item()} a:{a.item()} b:{b.item()}')
        print(f'{epoch + 1}/{num_epochs} loss:{loss.item()} a:{model.weight.item()} b:{model.bias.item()}')

#打印学到的参数
# a_learned = a.item()
# b_learned = b.item()
a_learned = model.weight.item()
b_learned = model.bias.item()
print("拟合的斜率 a:",{a_learned})
print("拟合的截距 b:",{b_learned})

# 禁用梯度计算，因为我们不再训练
with torch.no_grad():
    y_predicted = a_learned * x + b_learned

#绘制结果
# plt.figure(figsize=(10,6))
# plt.scatter(x_numpy,y_numpy,label='raw data',color='blue',alpha=0.6)#原始数据
# plt.plot(x_numpy,y_predicted,label=f'Model:y = {a_learned}x + {b_learned}',color = 'red',linewidth=2)#回归线
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend()
# plt.show()