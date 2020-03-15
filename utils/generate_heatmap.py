import numpy as np

def make_gaussian(size, center, is_ball=True, sigma=10):
    '''
    Parameters:
        size : tuple(h, w)
        center : tuple(h, w)
        is_ball : bool
    Returns:
        heatmap : np.array(size)
    '''
    if not is_ball:
        return np.zeros(size)
    y = np.arange(0, size[0], 1, np.float32)[:, np.newaxis]
    x = np.arange(0, size[1], 1, np.float32)
    y0, x0 = center
    return np.floor(np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) * 255)
