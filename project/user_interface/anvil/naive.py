import numpy as np
from scipy import linalg
import time as time
from sklearn.model_selection import train_test_split
from als_orig import yclassifier


def naive(file, split):
    start = time.process_time()
    train, test = train_test_split(file, test_size=split)
    cnames = train.columns.values
    p = 4
    y = yclassifier(train, p)
    # print(minv, maxv)

    res = [[0 for _ in range(p)] for _ in range(len(train))]
    for j in range(len(train)):
        flower = list(train.iloc[j, :-1])
        res[j] = [(flower[0]) ** k for k in range(p)]
        for i in range(1,len(cnames) - 1):
            # print(j,i, res[j])
            res[j] = np.outer(res[j],[(flower[i]) ** k for k in range(p)])
    print(f'shape: {np.asarray(res).shape}')
    res = np.reshape(np.asarray(res), (len(train), (len(cnames)-1)**p))
    print(res.shape)

    w = linalg.pinv(res).dot(y)
    yhat = res.dot(w)
    count = 0
    for i in range(len(yhat)):

        if yhat[i] * y[i] >= 0:
            count += 1
            # print(yhat[i], y[i])

    accuracy = np.round((count / len(yhat)) * 100, 2)
    stop = time.process_time()
    runtime = stop-start
    print(f'time = {runtime}, accuracy = {accuracy}')
    return accuracy, runtime