import numpy as np
import matplotlib.pyplot as plt

def ts_plot(x, color='blue', ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(range(len(x)), x, color=color)
    
    return ax
    
def dtw_plot(x, y, path, show_every=10, gap=0.1, ax=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(18, 8))
    
    for c, i, j in zip(range(len(path[0])), *path):
        if c % show_every == 0:
            ax.plot([i, j], [x[i]+gap, y[j]], color='r')
    
    ts_plot(x+gap, ax=ax)
    ts_plot(y, ax=ax)
    
    return ax
