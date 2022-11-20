# EmphysemaSeg
Automatic lobe segmentation and emphysema quantification. Input is a chest CT (.nii.gz) and outputs label maps for lobe segmentations and emphysema approximated as low attenuation volume (<-950HU). Visualization is also provided


## Citation
Please cite the following if you find this tool useful!

Thomas Z. Li, Ho Hin Lee, Kaiwen Xu, Riqiang Gao, Benoit M. Dawant, Fabien Maldonado, Kim L. Sandler, Bennett A. Landman. Quantifying Emphysema in Lung Screening Computed Tomography with Robust Automated Lobe Segmentation. Submitted to European Journal of Radiology on Nov. 2022.

## Usage with python
1. Download model weights from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7339893.svg)](https://doi.org/10.5281/zenodo.7339893)
2. Populate `config.YAML` with location of directories and model weights. See `example_config.YAML` for more details.
3. Running `EmphysemaSeg --config path_to_config.YAML` will only do lobe segmentation. Add `--emp` for emphysema segmentation and `--vis` for visualization.

## Limitations and Resource requirements
* This tool is designed for chest CTs with voxel dimensions in th range of 0.5-1mm x 0.5-1mm x 0.5-1.25mm (coronal x sagittal x axial). 
* Segmentation of emphysema with this tool is only reliable on soft kernel CT and NOT robust to hard kernel reconstructions.
  
| File size       | Runtime | RAM   | VRAM   |
|-----------------|---------|-------|--------|
| (512, 512, 400) | ~1m 20s  | <12 GB | 6.5 GB |