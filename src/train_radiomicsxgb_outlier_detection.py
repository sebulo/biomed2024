import vtk
import numpy as np
import os
from pathlib import Path
import argparse
from dtu_spine_config import DTUConfig
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle
from xgboost import XGBClassifier
from sklearn.utils import shuffle
from radiomics import featureextractor
import pandas as pd

def train_xgboost_model(settings):
    """
    Train an XGBoost model for outlier detection
    - We assume that the shapes are already aligned (no need for Procrustes)
    """
    print("Training XGBoost model")
    training_list = settings["data_set"]
    result_dir = settings["result_dir"]

    xgb_dir = os.path.join(result_dir, "radiomicsxgb_analysis")

    # Create folders if they don't exist
    Path(xgb_dir).mkdir(parents=True, exist_ok=True)
    
    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return
    
    # # Create Radiomics feature extractor
    # extractor = featureextractor.RadiomicsFeatureExtractor()
    # extractor.enableAllFeatures()

    # # Create holders
    # features = []
    # targets = []

    # # Now read all images and labels
    # for i, idx in enumerate(all_scan_ids):
    #     print(f"Reading {i + 1} / {len(all_scan_ids)}", flush=True)
    #     scan_id = idx.strip()

    #     imageName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop.nii.gz")
    #     maskName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_label.nii.gz")

    #     feature = extractor.execute(imageName, maskName, label=20)
    #     features.append(feature)
    #     targets.append(0)

    #     imageName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_warp_outlier.nii.gz")
    #     maskName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_label_warp_outlier.nii.gz")

    #     feature = extractor.execute(imageName, maskName, label=20)
    #     features.append(feature)
    #     targets.append(1)

    #     imageName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_sphere_outlier_water.nii.gz")
    #     maskName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_label_sphere_outlier_water.nii.gz")

    #     feature = extractor.execute(imageName, maskName, label=20)
    #     features.append(feature)
    #     targets.append(1)

    #     imageName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_sphere_outlier_mean_std_inpaint.nii.gz")
    #     maskName = os.path.join(data_dir, "train/crops", f"{scan_id}_crop_label_sphere_outlier_mean_std_inpaint.nii.gz")

    #     feature = extractor.execute(imageName, maskName, label=20)
    #     features.append(feature)
    #     targets.append(1)

    #     # if i == 3:
    #     #     break
        
    # features = pd.DataFrame.from_records(features)

    # # try to cast every column to float
    # for col in features.columns:
    #     try:
    #         features[col] = features[col].astype(float)
    #     except:
    #         pass
    
    # # Keep numerical features only
    # features = features.select_dtypes(include=[np.number])

    # # Export features as csv
    # features.to_csv(os.path.join(xgb_dir, "features.csv"), index=False)

    # Import radiomics features csv
    features = pd.read_csv(os.path.join(xgb_dir, "radiomics_train_feat.csv"))
    
    # Extract scan ids
    scan_ids = features["Image"].apply(lambda x: x[:11])

    # Keep matching scan ids only
    idx = scan_ids.isin(all_scan_ids)
    features = features[idx]

    # Determine targets
    targets = features["Image"].apply(lambda x: 1 if "_outlier" in x else 0)

    # # Debug print
    # for c in features.columns:
    #     print(c, features[c].dtype, features[c][0])

    # print(features.shape, list(features.columns), features)
    # print(targets)

    # Convert features to numpy array
    cols = list(features.columns)
    cols.remove("Image")
    cols.remove("Mask")
    data_matrix = features[cols].to_numpy()

    # Shuffle data
    data_matrix, targets = shuffle(data_matrix, targets, random_state=1000)

    # Train an XGBoost model
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst.fit(data_matrix, targets)

    # Export the model
    xgb_out = os.path.join(xgb_dir, "xgboost_model.pkl")
    print(f"Saving XGBoost model to {xgb_out}")
    with open(xgb_out, 'wb') as pickle_file:
        pickle.dump(bst, pickle_file)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='train-xgboost_outlier_detection')
    config = DTUConfig(args)
    if config.settings is not None:
        train_xgboost_model(config.settings)
