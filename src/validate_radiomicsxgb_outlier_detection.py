import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd

def validate_xgboost_outlier_detection(settings):
    """
    """
    print("Running PCA analysis on validation set")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    validation_results_json = os.path.join(result_dir, "validation_results.json")

    xgb_dir = os.path.join(result_dir, "radiomicsxgb_analysis")

    model_name = os.path.join(xgb_dir, "xgboost_model.pkl")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return
    
    # Import csv
    features = pd.read_csv(os.path.join(xgb_dir, "radiomics_train_feat.csv"))
    
    # Extract scan ids
    scan_ids = features["Image"].apply(lambda x: x[:11])

    # Keep matching scan ids only
    # print(all_scan_ids)
    # print([all_scan_ids[i][0].strip() for i in range(len(all_scan_ids))])
    # print(scan_ids)
    idx = scan_ids.isin([all_scan_ids[i][0].strip() for i in range(len(all_scan_ids))])
    # print(idx.sum(), idx)
    features = features[idx]

    # Determine targets
    targets = features["Image"].apply(lambda x: 1 if "_outlier" in x else 0)

    # # Debug print
    # for c in features.columns:
    #     print(c, features[c].dtype)

    # print(features.shape, list(features.columns), features)
    # print(targets)

    # Convert features to numpy array
    cols = list(features.columns)
    cols.remove("Image")
    cols.remove("Mask")
    data_matrix = features[cols].to_numpy()

    # Load the model
    with open(model_name, 'rb') as file:
        xgb_model = pickle.load(file)

    # Predict outliers
    outliers = xgb_model.predict(data_matrix)

    # Create classification results
    validation_results = []
    for i in range(all_scan_ids.shape[0]):
        scan_id = all_scan_ids[i][0].strip()
        # Remember to cast bools to int for json serialization
        validation_results.append({"scan_id": scan_id, "outlier": int(outliers[i]),
                                    #    "outlier_probability": float(outlier_probs[i]),
                                    #    "outlier_threshold": float(threshold)
                                       })

    # Write classification results to file
    with open(validation_results_json, 'w') as json_file:
        json.dump(validation_results, json_file, indent=4)

    # Print accuracy
    n_correct = np.sum(outliers == targets)
    print("Accuracy: ", n_correct / len(targets))


def compute_outlier_detection_metrics(settings):
    """
    """
    print("Computing outlier detection metrics")
    data_dir = settings["data_dir"]
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    validation_results_json = os.path.join(result_dir, "validation_results.json")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    n_samples = len(all_scan_ids)
    outliers_gt = np.zeros(n_samples, dtype=bool)
    outlier_pred = np.zeros(n_samples, dtype=bool)

    with open(validation_results_json, 'r') as json_file:
        validation_results = json.load(json_file)

    i = 0
    n_predicted_outliers = 0
    for idx in all_scan_ids:
        scan_id = idx[0].strip()
        outlier_type = idx[1].strip()
        if outlier_type != "":
            outliers_gt[i] = True

        for res in validation_results:
            if res["scan_id"] == scan_id:
                outlier_pred[i] = res["outlier"]
                n_predicted_outliers += 1
                break
        i += 1
    print(f"Found {n_predicted_outliers} predicted outliers out of {n_samples} samples")

    # Compute metrics
    tp = np.sum(outliers_gt & outlier_pred)
    tn = np.sum(~outliers_gt & ~outlier_pred)
    fp = np.sum(~outliers_gt & outlier_pred)
    fn = np.sum(outliers_gt & ~outlier_pred)
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn} N_pred: {n_predicted_outliers} N_samples: {n_samples}")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (tp + tn) / n_samples
    cohens_kappa = 2 * (tp * tn - fn * fp) / ((tp + fp) * (fp + tn) + (tp + fn) * (fn + tn))
    print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Accuracy: {accuracy:.2f}, "
          f"Cohens kappa: {cohens_kappa:.2f}")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='validate-pdm-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        validate_xgboost_outlier_detection(config.settings)
        compute_outlier_detection_metrics(config.settings)

