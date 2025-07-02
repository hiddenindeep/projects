import numpy as np

#计算矩阵的行列式
matrix = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(matrix)
print("行列式：", determinant)

#计算矩阵的逆矩阵
inverse_matrix = np.linalg.inv(matrix)
print("逆矩阵：", inverse_matrix)
print(np.dot(matrix, inverse_matrix))

#计算矩阵的特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(matrix)
print("特征值：", eigenvalues)
print("特征向量：", eigenvectors)

#解线性方程组
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 6])
solution = np.linalg.solve(A, b)
print("线性方程组的解：", solution)