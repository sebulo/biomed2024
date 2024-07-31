# Outlier Detection Challenge 2024
Outlier detection challenge 2024 - a DTU Compute summer school challenge

This challenge is made in conjunction with the [Ph. D. Summer school on biomedical image analysis – from acquisition to fairness and bias](https://biomed.compute.dtu.dk/). The Ph. D. summer school specific information can be found TBD.

The goal of this challenge is to develop and evaluate algorithms for outlier detection. In this challenge, outlier detection (sometimes called novelty detection) is to detect samples that are not considered *normal*.

Specifically, the challenge is focused on the human spine where we will look at the L1 vertebra as seen on a 3D computed tomography (CT) scan. A set of normal vertebra are given and a set with artificial artefacts (outliers) is also given. Finally, a test set with both normals and outliers are provided and the goal is to determine which samples in the test set that are outliers.

## Clinical background

The [spine](https://visualsonline.cancer.gov/details.cfm?imageid=12201) consists of a set of individual vertebra:
<img src="figs/nci-vol-12201-72.jpg" width=200/> 
<img src="figs/L1_top.jpg" width=200/> 

Three orthogonal slices of a CT scan with overlaid segmentation masks:

<img src="figs/3DSlicerView.jpg" width=200/> 

Volume rendering of a CT scan of the spine: 
<img src="figs/3DSlicerView_3.jpg" width=200/> 


## Data

For all samples there are different *representations* of the data. It is important to choose the representation that is best suited for the task at hand.

### Raw data and segmentation masks

A cropped CT scan with the original Hounsfield units and a segmentation mask, where the value of each voxel indicate if it is part of the vertebra or background (=0).

|        Raw data     |
|:----------------------------------------:|
|<img src="figs/3DSlicerView_2.jpg" width=600/>|


|        Raw data and segmentation mask     |
|:----------------------------------------:|
|<img src="figs/3DSlicerView.jpg" width=600/>|


|        Raw data and 3D rendering    |
|:----------------------------------------:|
|<img src="figs/3Dslicer_dist_segm.jpg" width=600/>|

### Distance fields

A distance field that is a 3D voxel volume, where the value in each voxel is the signed distance to the surface of the vertebra:

|                 Distance field                 |
|:----------------------------------------:|
|<img src="figs/3Dslicer_dist_field.jpg" width=600/>|


### Meshes

A mesh representing the outer surface of the vertebra:


|                 Mesh                 |                 Mesh triangles                 |                 Mesh vertices                 |
|:----------------------------------------:|:---------------------------------------------:|:-------------------------------------------:|
| <img src="figs/vertebra_mesh.jpg" width=200/> | <img src="figs/vertebra_mesh_2.jpg" width=200/> | <img src="figs/vertebra_mesh_3.jpg" width=200/> |

The meshes are pre-registered and there is *point-correspondence* over the entire data set. That means that all meshes have the same number of vertices and that a vertex with a given id can be assumed to be place on approximately the same anatomical location on all vertebra.

## Naming conventions






## Supplied Python scripts

## Dependencies

## Gettings started

## Configuring your submission

## Outlier detection evaluations

## The challenge score board

## Inspirations

### PCA based outlier detection

### Segmentation based outlier detection

### VAE based outlier detection
Single slice vs full volume




## Links and material

- [Scikit Learn on outlier and novelty detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

