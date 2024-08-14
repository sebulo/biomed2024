import vtk
import numpy as np
import os
import argparse
from dtu_spine_config import DTUConfig
import matplotlib.pyplot as plt
import pickle
import json
import pandas as pd


def test_xgboost_outlier_detection(settings):
    """
    """
    print("Running XGBoost analysis on test set")
    test_list = settings["data_set"]

    result_dir = settings["result_dir"]
    test_results_json = os.path.join(result_dir, "test_results.json")

    xgb_dir = os.path.join(result_dir, "radiomicsxgb_analysis")

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

    # Import csv
    features = pd.read_csv(os.path.join(xgb_dir, "radiomics_test_200_feat.csv"))
    
    # Extract scan ids
    scan_ids = features["Image"].apply(lambda x: x[:11])

    # Keep matching scan ids only
    idx = scan_ids.isin(all_scan_ids)
    features = features[idx]

    # # Debug print
    # for c in features.columns:
    #     print(c, features[c].dtype)

    # print(features.shape, list(features.columns), features)

    # Convert features to numpy array
    cols = list(features.columns)
    cols.remove("Image")
    cols.remove("Mask")
    data_matrix = features[cols].to_numpy()

    # Predict outliers
    outliers = xgb_model.predict(data_matrix)

    # Create results
    test_results = []
    for i in range(len(features["Image"].apply(lambda x: x[:11]))):
        scan_id = features["Image"][i][:11]
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
