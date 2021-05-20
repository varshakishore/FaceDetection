import numpy as np


G = np.array([[1, 1, 0, 1],
              [1, 0, 1, 1],
              [1, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]]).T

H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]]).T

H2 = np.array([1, 2, 4])

R = np.array([[0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 0, 1]]).T


class Hamming74(object):
    def __init__(self):
        pass

    def encode(self, x):
        # x: np.array, (kx4,)
        # output: np.array, (kx7,)
        return (x.reshape(-1, 4) @ G).reshape(-1) % 2

    def decode(self, x):
        # x: np.array, (kx7,)
        # output: np.array, (kx4,)
        x = x.reshape(-1, 7)
        z = ((x @ H) % 2) @ H2
        xx = np.where(z > 0)[0]
        yy = z[xx] - 1
        x[xx, yy] = 1 - x[xx, yy]
        return (x[:, [2, 4, 5, 6]]).reshape(-1) # equivalent to return (x @ R).reshape(-1)


if __name__ == "__main__":
    hamming = Hamming74()
    x = np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    print(x)
    y = hamming.encode(x)
    print(y)
    x[[1, 6, 11, 12]] = -x[[1, 6, 11, 12]]
    z = hamming.decode(y)
    print(z)

