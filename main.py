import matplotlib.pyplot as plt
import numpy as np

vecN = np.random.randint(0, 10, size=10)
vecF = np.random.rand(10)
vec3 = np.random.randint(0, 10, 5) * 3

print(vecN)


def fibo_list(size):
    res = [1, 1]
    for i in range(size - 2):
        res.append(res[i] + res[i + 1])
    return np.array(res)


def showCharts():
    ax = plt.gca()  # gca stands for 'get current axis'
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    plt.plot(x, h1)
    # plt.plot(x, y2)
    # plt.plot(x1, third_array)
    plt.scatter(x, first_array, c='r')
    # plt.scatter(x, second_array, c='y')
    plt.show()


fibo = fibo_list(10)[np.random.randint(0, 10, size=10)]
print(fibo)

x = np.linspace(-50, 50, 10)
y1 = x * 2
first_array = y1 + np.random.normal(size=10)

y2 = x * 5 + 70
np.random.normal()
second_array = y2 + np.random.normal(size=10)

x1 = np.linspace(-50, 50, 20)
third_array = x1 ** 2

mat1 = np.random.randn(4, 4)
mat2 = np.random.randint(0, 100, 16).reshape((4, 4))
mat3 = np.dot(mat1, mat2)
mat4 = np.linalg.inv(mat3)
mat5 = np.transpose(mat3)
# showCharts()




X = np.column_stack((np.ones(10), second_array))
teta = np.dot(np.linalg.inv(np.dot(X.transpose(), X)), np.dot(X.transpose(), y2))
print(teta)

x2 = np.array([np.ones(10), x]).transpose()
h1 = np.dot(x2, teta)
showCharts()
