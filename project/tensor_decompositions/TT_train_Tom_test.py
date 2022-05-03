import numpy as np
from scipy import linalg
from PIL import Image

dog = np.asarray(Image.open('dog.jpg'))
x = np.reshape(dog, (4, 4, 4, 4, 4, 4, 4, 4, 4, 3))
threshold = 0
max_rank = np.infty

# define order, row dimensions, column dimensions, ranks, and cores
order = len(x.shape) // 2
row_dims = x.shape[:order]
col_dims = x.shape[order:]
ranks = [1] * (order + 1)
cores = []

# permute dimensions, e.g., for order = 4: p = [0, 4, 1, 5, 2, 6, 3, 7]
p = [order * j + i for i in range(order) for j in range(2)]
y = np.transpose(x, p).copy()

# decompose the full tensor
for i in range(order - 1):
    # reshape residual tensor
    m = ranks[i] * row_dims[i] * col_dims[i]
    n = np.prod(row_dims[i + 1:]) * np.prod(col_dims[i + 1:])
    y = np.reshape(y, [m, n])

    # apply SVD in order to isolate modes
    [u, s, v] = linalg.svd(y, full_matrices=False)

    # rank reduction
    if threshold != 0:
        indices = np.where(s / s[0] > threshold)[0]
        u = u[:, indices]
        s = s[indices]
        v = v[indices, :]
    if max_rank != np.infty:
        u = u[:, :np.minimum(u.shape[1], max_rank)]
        s = s[:np.minimum(s.shape[0], max_rank)]
        v = v[:np.minimum(v.shape[0], max_rank), :]

    # define new TT core
    ranks[i + 1] = u.shape[1]
    cores.append(np.reshape(u, [ranks[i], row_dims[i], col_dims[i], ranks[i + 1]]))

    # set new residual tensor
    y = np.diag(s).dot(v)

# define last TT core
cores.append(np.reshape(y, [ranks[-2], row_dims[-1], col_dims[-1], 1]))

print("\n"
      "Tensor train with order    = {d}, \n"
      "                  row_dims = {m}, \n"
      "                  col_dims = {n}, \n"
      "                  ranks    = {r}".format(d=order, m=row_dims, n=col_dims, r=ranks))
