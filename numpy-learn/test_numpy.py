import numpy as np
#提供高维数组对象numpy.ndarray,运算速度碾压python原生list
#提供矩阵运算，向量运算，随机数生成，傅里叶变换，线性代数，统计，科学计算等

#ndarray属性：
#ndarray.ndim:数组的维度
#ndarray.shape:数组的形状
#ndarray.size:数组元素的总个数
#ndarray.dtype:数组元素的类型

#强制类型统一：numpy设计的初衷就是为了处理数值运算，所以它要求数组中的元素类型必须统一 优先级：str>float>int

a1 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(a1)

# order='C'
# 指定内存布局方式
# 'C'：行优先（C风格），按行存储
# 'F'：列优先（Fortran风格），按列存储

#用1创建二维数组
a2 = np.ones(shape=(3,4),dtype=np.int32,order='C')
print(a2)

#用0创建二维数组
a3 = np.zeros(shape=(3,4),dtype=np.int32,order='C')
print(a3)

#用指定元素创建数组
a4 = np.full(shape=(3,4),fill_value=6.89,dtype=np.float16,order='C')
print(a4)

#eye:创建单位矩阵
a5 = np.eye(N=3,M=3,k=0,dtype=np.float16,order='C')
print(a5)

#linspace:创建等差数列,指定个数
a6 = np.linspace(start=0,stop=10,num=5,endpoint=False,retstep=False,dtype=int)
print(a6)

#arange:创建等差数列,指定步长
a7 = np.arange(start=0,stop=10,step=2,dtype=int)
print(a7)

#普通正态分布:loc:均值,scale:标准差,size:形状
a8 = np.random.normal(loc=170,scale=10,size=(3,3))
print(a8)

#标准正态分布:均值为0,标准差为1
a9 = np.random.randn(3,3)
print(a9)

#[0,1)随机浮点数
a10 = np.random.random(size=(3,3))
print(a10)

#随机种子 相同种子=相同序列 无论何时何地运行相同种子总是产生相同的随机数序列
np.random.seed(10)
a11 = np.random.randint(low=-5,high=5,size=(3,4))
print(a11)

a12 = np.linspace(start=0,stop=2*np.pi,num=8,endpoint=False,retstep=False,dtype=np.float64)
print(a12)
print(np.degrees(a12))

#索引访问
arr5 = np.random.randint(low=-5,high=5,size=(3,4))
print(arr5)
print(arr5[2][3]) #间接访问：先访问行索引，再访问列索引，不建议使用
print(arr5[2,3])
#列表访问
index1 = [0,2]
print(arr5[index1])
#切片访问
print(arr5[1:2,1:3])

#ndarray[dim1_index,dim2_index,...dimn_index]
#dim_index:整数、切片、整数数组、布尔数组