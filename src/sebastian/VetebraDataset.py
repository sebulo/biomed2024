import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import vtk
from vtk.util.numpy_support import vtk_to_numpy

class VertebraDataset(Dataset):
    def __init__(self, data_dir, file_list, transform=None, data_type='image'):
        """
        Args:
            data_dir (str): Directory with all the data files.
            file_list (list): List of file identifiers (e.g., ['sample_0017', 'sample_0018', ...]).
            transform (callable, optional): Optional transform to be applied on a sample.
            data_type (str): The type of data to load: 'image', 'mesh', or 'segmentation'.
        """
        self.data_dir = data_dir
        self.file_list = file_list
        self.transform = transform
        self.data_type = data_type

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_id = self.file_list[idx].strip()

        if self.data_type == 'image':
            # Load image data (assuming .png format)
            img_path = os.path.join(self.data_dir, f"{sample_id}_image.png")
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image

        elif self.data_type == 'mesh':
            # Load VTK mesh data
            mesh_path = os.path.join(self.data_dir, f"{sample_id}_surface.vtk")
            mesh_data = self.load_vtk_mesh(mesh_path)
            return torch.tensor(mesh_data, dtype=torch.float32)

        elif self.data_type == 'segmentation':
            # Load segmentation data (assuming .nii.gz format)
            seg_path = os.path.join(self.data_dir, f"{sample_id}_segmentation.nii.gz")
            segmentation_data = self.load_nifti_file(seg_path)
            return torch.tensor(segmentation_data, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def load_vtk_mesh(self, file_path):
        """ Load VTK mesh and convert to a numpy array. """
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()
        polydata = reader.GetOutput()
        points = polydata.GetPoints()
        numpy_points = vtk_to_numpy(points.GetData())
        return numpy_points

    def load_nifti_file(self, file_path):
        """ Load NIfTI file (3D segmentation) and convert to a numpy array. """
        import nibabel as nib
        img = nib.load(file_path)
        return img.get_fdata()