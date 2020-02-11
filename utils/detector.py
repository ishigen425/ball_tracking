import numpy as np

def detect(heatmap):
    heatmap_thread = np.where(heatmap < 127, 0, heatmap)
    return np.unravel_index(np.argmax(heatmap_thread), heatmap_thread.shape)

def judge(target_heatmap, output_heatmap, sigma=10):
    max_y, max_x = detect(output_heatmap)
    if max_y + max_x == 0:
        return 0
    tar_y, tar_x = detect(target_heatmap)
    dist = np.sqrt((tar_x-max_x) ** 2 + (tar_y-max_y) ** 2)
    if dist > sigma:
        return 0
    return 1
