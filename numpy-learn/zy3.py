import numpy as np

a1 = np.random.randint(1,20,size=(4,4))
a2 = np.random.randint(1,20,size=(8,4))

a3 = np.concatenate((a1,a2),axis=0)
print(a3)

#拆成3等份
a4,a5,a6 = np.split(a3,[4,8],axis=0)
# print(a4)
# print(a5)
# print(a6)

#12可以被3整除，直接纵向3等分
a7,a8,a9 = np.vsplit(a3,3)
print(a7)
print(a8)
print(a9)

a10 = np.random.randint(1,20,(5,))
print(a10.dtype)
#astype():ndarray实例方法，将每一个元素转换
a11 = a10.astype(np.float32)
print(a11.dtype)