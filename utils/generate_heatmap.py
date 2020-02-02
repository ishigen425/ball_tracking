import numpy as np

def gaussian(x, y, x0, y0, sigma=10):
    sigma_d = sigma ** 2
    t0 = np.exp(-((x-x0) ** 2 + (y-y0) ** 2) / (2 * sigma_d))
    first = t0 / (2 * np.pi * sigma_d)
    second = 2 * np.pi * sigma_d * 255
    return np.floor(first * second)

def make_gaussian(size, center, is_ball=True):
    '''
    Parameters:
        size : tuple(x, y)
        center : tuple(x, y)
        is_ball : bool
    Returns:
        heatmap : np.array(size)
    '''
    heatmap = np.zeros(size)
    if not is_ball:
        return heatmap
    x, y = size
    for i in range(x):
        for j in range(y):
            heatmap[i, j] = gaussian(i, j, *center)
    return heatmap
