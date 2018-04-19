import numpy as np

default_lookup_list = lambda i, j: [
    (i-1, j-1),
    (i-1, j),
    (i, j-1),
]

euclidian_dist = lambda x, y: (x-y)**2

def dtw_path(D, lookup_list=default_lookup_list):
    i, j = np.array(D.shape) - 1
    p, q = [i], [j]
    while ((i > 0) or (j > 0)):
        next_ij = lookup_list(i, j)
        
        tb = np.argmin([D[a, b] for a, b in next_ij])
        i, j = next_ij[tb]
        if i <= 0:
            i = 0
        if j <= 0:
            j = 0
        
        p.append(i)
        q.append(j)
    return np.array(list(reversed(p))), np.array(list(reversed(q)))

def dtw(x, y, w=0.10, dist=euclidian_dist, lookup_list=default_lookup_list, return_all=False):
    r, c = len(x), len(y)
    D = np.ones((r, c))*np.inf
    
    k = int(r*w)
    
    for i in range(r):
        for j in range(max(0, i-k), min(c, i+k+1)):
            D[i, j] = dist(x[i], y[j])
            
            min_d = np.inf
            for a, b in lookup_list(i, j):
                if 0 <= a < r and 0 <= b < c:
                    min_d = min(min_d, D[a, b])

            if min_d != np.inf:
                D[i, j] += min_d
            
    path = dtw_path(D, lookup_list)
        
    dist = D[-1, -1] / sum(D.shape)
    
    if return_all:
        return dist, path, D
    
    return dist, path