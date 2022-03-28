import numpy as np

# 整数列表转转换为numpy数组
a1 = [1, 2, 3]
a2 = np.array(a1)
# 查看数据类型
print(type(a1))
print(type(a2))

# 浮点数列表转换为numpy数组
b = np.array([1.2, 2.3, 3.4])
print(b.dtype)

# 将两个整数列表转换为二维numpy数组
c = np.array([[1, 2, 3], [4, 5, 6]])
print(c)

# 生成2*3的全0数组
# d1 = np.zeros((2, 3))
d1 = np.empty((2, 3))
print(d1)
# 生成3*4全1数组,类型为int16
d2 = np.ones((3, 4), dtype=np.int16)
print(d2)
print(d2.dtype)
# 生成指定范围和步长的数组
d3 = np.arange(10, 30, 5)
print(d3)
# 生成2*3的随机数数组
d4 = np.random.rand(3, 2)
print(d4)
# 生成0-2范围内个数为5的数组
d5 = np.random.randint(3, size=5)
print(d5)
# 生成一组符合正太分布的随机数数组
d6 = np.random.randn(3)
print(d6)

# 创建一个一维数组
e1 = np.arange(10) ** 2
print(e1)
# 获取数组的第3个元素
e2 = e1[2]
print(e2)
# 获取第2-4个数组元素
e3 = e1[1:4]
print(e3)
# 一维数组翻转
e4 = e1[::-1]
print(e4)

# 创建一个多维数组
f1 = np.random.random((3, 3))
print(f1)
# 获取第2行第3列的数组元素
f2 = f1[1, 2]
print(f2)
# 获取第2列数组
f3 = f1[:, 1]
print(f3)
# 获取第3列前两行的数据
f4 = f1[:2, 2]
print(f4)

# 创建两个不同的数组
g1 = np.arange(4)
g2 = np.array([5, 10, 15, 20])
# 两个数组做减法运算
print(g2 - g1)
# 计算数组的平方
print(g2 ** 2)
# 计算数组的正弦值
print(np.sin(g1))
# 数组的逻辑运算
print(g2 < 20)
# 数组求均值和方差
print(np.mean(g2))
print(np.var(g2))

# 创建连个不同的数组
h1 = np.array([[1, 1], [0, 1]])
h2 = np.array([[2, 0], [3, 4]])
# 矩阵元素乘积
h3 = h1 * h2
print(h3)
# 矩阵点乘
h4 = h1.dot(h2)
print(h4)
# 矩阵求逆
h5 = np.linalg.inv(h1)
print(h5)
# 矩阵求行列式
h6 = np.linalg.det(h1)
print(h6)
# 按列合并数组h1和h2
h7 = np.vstack((h1, h2))
print(h7)
# 按行合并数组h1和h2
h8 = np.hstack((h1, h2))
print(h8)

# 创建一个3*4的数组
j1 = np.floor(10 * np.random.random((3, 4)))
print(j1)
# 查看数组维度
print(j1.shape)
# 数组展平
print(j1.ravel())
# 将数组变换为2*6数组
j2 = j1.reshape(2, 6)
print(j2)
# 求数组的转置
j3 = j1.T
print(j3)
print(j3.shape)
# -1维度表示numpy会自动计算该维度
print(j1.reshape(4, -1))

# 创建一个新数组
k1 = np.arange(16.0).reshape(4, 4)
print(k1)
# 按水平方向将数组切分为两个数组
print(np.hsplit(k1, 2))
# 按垂直方向将数组切分为两个数组
print(np.vsplit(k1, 2))

