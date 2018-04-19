import numpy as np

default_lookup = lambda D, i, j: min(D[i, j], D[i, j+1], D[i+1, j])
euclidian_dist = lambda x, y: (x-y)**2

def dtw_path(D):
    i, j = np.array(D.shape) - 2
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        tb = np.argmin((D[i, j], D[i, j+1], D[i+1, j]))
        if (tb == 0):
            i -= 1
            j -= 1
        elif (tb == 1):
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)

def dtw(x, y, dist=euclidian_dist, lookup=default_lookup, return_all=False):
    r, c = len(x), len(y)
    D0 = np.zeros((r + 1, c + 1))
    D0[0, 1:] = np.inf
    D0[1:, 0] = np.inf
    D1 = D0[1:, 1:]
    for i in range(r):
        for j in range(c):
            D1[i, j] = dist(x[i], y[j])
            
    C = D1.copy()
    for i in range(r):
        for j in range(c):
            D1[i, j] += lookup(D0, i, j)
            
    if len(x)==1:
        path = np.zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), np.zeros(len(x))
    else:
        path = dtw_path(D0)
        
    dist = D1[-1, -1] / sum(D1.shape)
    
#     return dist, C, D1, path
    return dist, path
