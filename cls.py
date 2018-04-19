import numpy as np
from dtw import dtw

def nn_dtw_cls(Q, X, y):
    ans = list()
    for q in Q:
        min_index = np.argmin([dtw(q, x, w=0.05)[0] for x in X])
        ans.append(y[min_index])
    return np.array(ans)
