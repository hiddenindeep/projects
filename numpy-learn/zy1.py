import numpy as np

#创建一个形状为（3，4）的二维数组，取值范围为（-5，5）
arr = np.random.randint(low=-5,high=5,size=(3,4))
print(arr)

# 创建一个包含5个元素的数组，元素从0开始，到15结束，但不包括15，数据类型为int
arr1 = np.linspace(start=0,stop=15,num=5,endpoint=False,retstep=False,dtype=int)
print(arr1)

# 创建一个包含从0开始，到15结束，步长为3的数组，数据类型为int
arr2 = np.arange(start=0,stop=15,step=3,dtype=int)
print(arr2)

# 创建一个100x100x3的随机数组
arr3 = np.random.random(size=(100,100,3))
#print(arr3)

#计算出将一个圆等分成8份的弧度的代码
arr4 = np.linspace(start=0,stop=2*np.pi,num=8,endpoint=False)
print(arr4)
step = 2 * np.pi / 8
fd = np.degrees(step)
print('角度：%f 弧度:%f' % (step, fd))