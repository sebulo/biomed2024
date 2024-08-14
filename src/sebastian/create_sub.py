# create submissions
import json
import numpy as np

if __name__=="__main__":
    probs = np.load('prob.npy')
    with open('../results/test_files_200.txt', 'r') as f:
        lines = f.read()
    lines = lines.split('\n')[:-1];
    ds = []
    for i, line in enumerate(lines, 0):
        prob = probs[i];
        d = {
        "scan_id": line,
        "outlier": int(prob>0.15),
        "outlier_probability": str(prob),
        "outlier_threshold": str(0.15)
        }  
        ds.append(d)
    with open('../results/test_results.json', 'w') as f:
        json.dump(ds, f)
