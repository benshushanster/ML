# import matplotlib.pyplot as plt
import numpy as np

# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

#
# XX = np.array([[31, 22],
#                [22, 21],
#                [40, 37],
#                [26, 25]], dtype=np.float32)
# y = np.array([2, 3, 8, 12])
#
# W = np.linalg.inv(np.transpose(XX) @ XX) @ np.transpose(XX) @ y

theta = np.array([2, 2, 0], dtype=np.float32)
y = [1, 3, 7]
XX = np.array([[1, 0, 0],
               [1, 1, 1],
               [1, 2, 4]], dtype=np.float32)

for epoch in range(1000):
    h = XX @ theta
    loss = np.sum((h - y) ** 2)
    grad = (h - y) @ XX
    theta -= 0.01 * grad
    print("epoch:{} loss:{}".format(epoch, loss))
