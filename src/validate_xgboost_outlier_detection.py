import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json


def validate_xgboost_outlier_detection(settings):
    """
    """
    print("Running PCA analysis on validation set")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "train/surfaces")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    validation_results_json = os.path.join(result_dir, "validation_results.json")

    xgb_dir = os.path.join(result_dir, "xgboost_analysis")

    model_name = os.path.join(xgb_dir, "xgboost_model.pkl")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Read the mean mesh to determine the number of points
    # we also keep it for later use - to synthesize shapes
    # id_0 = all_scan_ids[0].strip()
    # surf_name = os.path.join(surface_dir, f"{id_0}_surface.vtk")
    id_0 = all_scan_ids[0][0].strip()
    surf_name = os.path.join(surface_dir, f"{id_0}_surface.vtk")
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(surf_name)
    reader.Update()
    first_surface = reader.GetOutput()

    n_points = first_surface.GetNumberOfPoints()
    # Three features per point (x, y, z)
    n_features = n_points * 3
    n_samples = len(all_scan_ids)
    print(f"Creating date matrix of size {n_samples} x {n_features}")
    data_matrix = np.zeros((n_samples, n_features))

    # Now read all meshes
    i = 0
    outliers_gt = np.zeros(n_samples, dtype=bool)
    for idx in all_scan_ids:
        scan_id = idx[0].strip()

        outlier_type = idx[1].strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface{outlier_type}.vtk")
        print(f"Reading {i + 1} / {n_samples} : {surf_name}")
        if outlier_type != "":
            outliers_gt[i] = True

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(surf_name)
        reader.Update()
        surface = reader.GetOutput()
        n_p = surface.GetNumberOfPoints()
        if n_p != n_points:
            print(f"Number of points in {scan_id} is {n_p} and it should be {n_points}")
            return
        for j in range(n_p):
            p = surface.GetPoint(j)
            data_matrix[i, j * 3] = p[0]
            data_matrix[i, j * 3 + 1] = p[1]
            data_matrix[i, j * 3 + 2] = p[2]
        i += 1

    # Load the model
    with open(model_name, 'rb') as file:
        xgb_model = pickle.load(file)

    # Predict outliers
    outlier_probs = xgb_model.predict_proba(data_matrix)[:, 1]

    # Find outliers by guessing that 25% of the samples are outliers
    # Start by finding the .75 percentile of the outlier probabilities
    threshold = np.percentile(outlier_probs, 75)
    print(f"Threshold is {threshold:.1f}")

    # Select the 25% most probable outliers
    outliers = outlier_probs >= threshold
    print(f"Found {np.sum(outliers)} outliers")
    print(f"Outliers: {all_scan_ids[outliers]}")
    print(f"Non-outliers: {all_scan_ids[~outliers]}")
    print(f"Outlier probs: {outlier_probs[outliers]}")
    print(f"Non-outlier probs: {outlier_probs[~outliers]}")
    print(f"Outlier probs mean: {np.mean(outlier_probs[outliers])}")
    print(f"Non-outlier probs mean: {np.mean(outlier_probs[~outliers])}")
    print(f"Outlier probs std: {np.std(outlier_probs[outliers])}")
    print(f"Non-outlier probs std: {np.std(outlier_probs[~outliers])}")
    print(f"Outlier probs min: {np.min(outlier_probs[outliers])}")
    print(f"Non-outlier probs min: {np.min(outlier_probs[~outliers])}")
    print(f"Outlier probs max: {np.max(outlier_probs[outliers])}")
    print(f"Non-outlier probs max: {np.max(outlier_probs[~outliers])}")

    # Create classification results
    validation_results = []
    for i in range(n_samples):
        scan_id = all_scan_ids[i][0].strip()
        # Remember to cast bools to int for json serialization
        validation_results.append({"scan_id": scan_id, "outlier": int(outliers[i]),
                                       "outlier_probability": float(outlier_probs[i]),
                                       "outlier_threshold": float(threshold)})

    # Write classification results to file
    with open(validation_results_json, 'w') as json_file:
        json.dump(validation_results, json_file, indent=4)


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

