import numpy as np


def mesh3d(x, y, z, dtype=np.float32): # x = [x1,x2,x3,..]; y =y1,y2,y3..], z=[z1,z2,...]
    grid = np.empty(x.shape + y.shape + z.shape + (3,), dtype=dtype) # grid: 6 x 6 x 6 x 3
    grid[..., 0] = x[:, np.newaxis, np.newaxis]
    grid[..., 1] = y[np.newaxis, :, np.newaxis]
    grid[..., 2] = z[np.newaxis, np.newaxis, :]
    return grid

def extent(x, *args, **kwargs):
    return np.min(x, *args, **kwargs), np.max(x, *args, **kwargs)

