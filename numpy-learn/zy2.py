#ndarray[dim1_index,dim2_index,...dimn_index]
#dim_index:整数、切片、整数数组、布尔数组

import numpy as np

arr = np.random.randint(1,100,(5,5))
#print(arr)

#取第1,2列的元素
# print(arr[:,[1,2]])

# #取第3行，第4，5列的元素
# print(arr[2,[3,4]])

bool_arr = [True,False,True,False,True]
#print(arr[bool_arr])

#使用布尔数组取第3行，第1，3，5列的元素
#print(arr[2,bool_arr])

#逆序输出
arr1 = np.random.randint(1,100,(10,))
print(arr1)
print(arr1[::-1])

#取最后两列
arr2 = np.random.randint(1,100,(5,4))
print(arr2)
print(arr2[:,[2,3]])
print(arr2[:,-2:])

bool_arr1 = [False,False,True,True]
print(arr2[:,bool_arr1])
