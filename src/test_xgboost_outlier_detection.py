import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json


def test_xgboost_outlier_detection(settings):
    """
    """
    print("Running XGBoost analysis on test set")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "test/surfaces")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    test_results_json = os.path.join(result_dir, "test_results.json")

    xgb_dir = os.path.join(result_dir, "xgboost_analysis")

    test_id_list_file = os.path.join(result_dir, test_list)
    all_scan_ids = np.loadtxt(str(test_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} test samples in {test_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Load XGBoost model
    model_name = os.path.join(xgb_dir, "xgboost_model.pkl")
    with open(model_name, 'rb') as picklefile:
        xgb_model = pickle.load(picklefile)
    
    # Read first mesh to determine the number of points
    # we also keep it for later use - to synthesize shapes
    # id_0 = all_scan_ids[0].strip()
    # surf_name = os.path.join(surface_dir, f"{id_0}_surface.vtk")
    id_0 = all_scan_ids[0].strip()
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
    for idx in all_scan_ids:
        scan_id = idx.strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface.vtk")
        print(f"Reading {i + 1} / {n_samples} : {surf_name}")

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

    # # Predict outliers
    # outlier_probs = xgb_model.predict_proba(data_matrix)[:, 1]

    # # Find outliers by guessing that 50% of the samples are outliers
    # # Start by finding the .5 percentile of the outlier probabilities
    # threshold = np.percentile(outlier_probs, 50)
    # print(f"Threshold is {threshold:.1f}")

    # # Select the 25% most probable outliers
    # outliers = outlier_probs >= threshold

    # Predict outliers
    outliers = xgb_model.predict(data_matrix)

    # Create results
    test_results = []
    for i in range(n_samples):
        scan_id = all_scan_ids[i].strip()
        # Remember to cast bools to int for json serialization
        test_results.append({"scan_id": scan_id, "outlier": int(outliers[i]),
                                    #    "outlier_probability": float(outlier_probs[i]),
                                    #    "outlier_threshold": float(threshold)
                                    })

    # Write results to JSON file
    with open(test_results_json, 'w') as json_file:
        json.dump(test_results, json_file, indent=4)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test-pdm-outlier-detection')
    config = DTUConfig(args)
    if config.settings is not None:
        test_xgboost_outlier_detection(config.settings)
