import numpy as np

# 示例 1: 数组创建函数
# 使用 numpy 的各种数组创建函数
zeros_array = np.zeros((3, 4))
ones_array = np.ones((2, 3))
eye_matrix = np.eye(3)
arange_array = np.arange(0, 10, 2)
linspace_array = np.linspace(0, 1, 5)

print("零数组:\n", zeros_array)
print("全1数组:\n", ones_array)
print("单位矩阵:\n", eye_matrix)
print("等差数组:", arange_array)
print("线性空间数组:", linspace_array)

# 示例 2: 数组条件筛选
# 使用条件筛选和布尔索引
data = np.array([1, 5, 3, 8, 2, 9, 4])
condition = data > 4
filtered_data = data[condition]
where_result = np.where(data > 4, data, 0)

print("原始数据:", data)
print("条件筛选结果:", filtered_data)
print("where函数结果:", where_result)

# 示例 3: 统计函数
# 使用各种统计函数
matrix = np.random.randint(1, 10, (3, 4))
print("随机矩阵:\n", matrix)
print("标准差:", np.std(matrix))
print("方差:", np.var(matrix))
print("中位数:", np.median(matrix))
print("分位数:", np.percentile(matrix, 75))

# 示例 4: 数组排序和搜索
# 排序和搜索相关函数
unsorted_array = np.array([3, 1, 4, 1, 5, 9, 2, 6])
sorted_array = np.sort(unsorted_array)
argsort_indices = np.argsort(unsorted_array)
unique_values = np.unique(unsorted_array)

print("原始数组:", unsorted_array)
print("排序后数组:", sorted_array)
print("排序索引:", argsort_indices)
print("唯一值:", unique_values)

# 示例 5: 数组拼接和分割
# 数组的合并和分割操作
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
concatenated = np.concatenate([arr1, arr2], axis=0)
stacked_h = np.hstack([arr1, arr2])
stacked_v = np.vstack([arr1, arr2])

print("数组1:\n", arr1)
print("数组2:\n", arr2)
print("垂直拼接:\n", concatenated)
print("水平堆叠:\n", stacked_h)
print("垂直堆叠:\n", stacked_v)

# 示例 6: 随机模块
# numpy.random 模块的重要函数
np.random.seed(42)  # 设置随机种子
random_normal = np.random.normal(0, 1, 10)
random_uniform = np.random.uniform(0, 1, 5)
random_choice = np.random.choice([1, 2, 3, 4, 5], 3)

print("正态分布随机数:", random_normal)
print("均匀分布随机数:", random_uniform)
print("随机选择:", random_choice)

# 示例 7: 数组比较和逻辑运算
# 逻辑运算函数
arr_a = np.array([1, 2, 3, 4, 5])
arr_b = np.array([3, 2, 1, 4, 6])
equal_result = np.equal(arr_a, arr_b)
greater_result = np.greater(arr_a, arr_b)
logical_and = np.logical_and(arr_a > 2, arr_b < 5)

print("数组A:", arr_a)
print("数组B:", arr_b)
print("相等比较:", equal_result)
print("大于比较:", greater_result)
print("逻辑与运算:", logical_and)

# 示例 8: 矩阵运算
# 线性代数相关函数
matrix = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(matrix)
inverse_matrix = np.linalg.inv(matrix)
eigenvalues, eigenvectors = np.linalg.eig(matrix)

print("原始矩阵:\n", matrix)
print("行列式:", determinant)
print("逆矩阵:\n", inverse_matrix)
print("特征值:", eigenvalues)
print("特征向量:\n", eigenvectors)
