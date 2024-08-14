# create submissions
import json
import numpy as np

if __name__=="__main__":
    probs = np.load('prob.npy')
    with open('../results/test_files_200.txt', 'r') as f:
        lines = f.read()
    lines = lines.split('\n')[:-1];
    for i, line in enumerate(lines, 0):
        prob = probs[i];
        d = {
        "scan_id": lines,
        "outlier": 0,
        "outlier_probability": 0.29324252683460816,
        "outlier_threshold": 0.392965204318129
    },
