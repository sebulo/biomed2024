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


def train_xgboost_model(settings):
    """
    Train an XGBoost model for outlier detection
    - We assume that the shapes are already aligned (no need for Procrustes)
    """
    print("Training XGBoost model")
    data_dir = settings["data_dir"]
    surface_dir = os.path.join(data_dir, "train/surfaces")
    training_list = settings["data_set"]
    result_dir = settings["result_dir"]

    xgb_dir = os.path.join(result_dir, "xgboost_analysis")

    # Create folders if they don't exist
    Path(xgb_dir).mkdir(parents=True, exist_ok=True)
    
    training_id_list_file = os.path.join(result_dir, training_list)
    all_scan_ids = np.loadtxt(str(training_id_list_file), delimiter=",", dtype=str)
    print(f"Found {len(all_scan_ids)} samples in {training_id_list_file}")
    if len(all_scan_ids) == 0:
        print(f"No samples found")
        return

    # Read first mesh to determine the number of points
    # we also keep it for later use - to synthesize shapes
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
    data_matrix = np.zeros((n_samples*4, n_features))

    target = np.zeros(n_samples*4)

    # Now read all meshes
    i = 0
    for idx in all_scan_ids:
        print(f"Reading {i + 1} / {n_samples}")
        scan_id = idx.strip()
        surf_name = os.path.join(surface_dir, f"{scan_id}_surface.vtk")
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
            data_matrix[i*4, j*3] = p[0]
            data_matrix[i*4, j*3+1] = p[1]
            data_matrix[i*4, j*3+2] = p[2]

        surf_name = os.path.join(surface_dir, f"{scan_id}_surface_warp_outlier.vtk")
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
            data_matrix[i*4+1, j*3] = p[0]
            data_matrix[i*4+1, j*3+1] = p[1]
            data_matrix[i*4+1, j*3+2] = p[2]

        surf_name = os.path.join(surface_dir, f"{scan_id}_surface_sphere_outlier_water.vtk")
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
            data_matrix[i*4+2, j*3] = p[0]
            data_matrix[i*4+2, j*3+1] = p[1]
            data_matrix[i*4+2, j*3+2] = p[2]

        surf_name = os.path.join(surface_dir, f"{scan_id}_surface_sphere_outlier_mean_std_inpaint.vtk")
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
            data_matrix[i*4+3, j*3] = p[0]
            data_matrix[i*4+3, j*3+1] = p[1]
            data_matrix[i*4+3, j*3+2] = p[2]
        
        target[i*4] = 0
        target[i*4+1] = 1
        target[i*4+2] = 1
        target[i*4+3] = 1        

        i += 1

    # Shuffle data
    data_matrix, target = shuffle(data_matrix, target, random_state=1000)

    # Train an XGBoost model
    bst = XGBClassifier(n_estimators=2, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst.fit(data_matrix, target)

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
