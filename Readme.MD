# VASCilia: A Revolutionary Napari Plugin for Cochlear StereoCilia Analysis
 
 Embark on a groundbreaking journey through the intricacies of the cochlea with VASCilia, the pioneering Napari plugin designed for the 3D segmentation and quantification of stereocilia bundles. Unveiling a suite of powerful features, VASCilia is a beacon of innovation for the auditory research realm, offering:

1. Selective Slice Discovery: Traverse the 3D landscape to pinpoint pivotal slices within your stack.
2. Stack Rotation: Align your perspective with precision for an optimal analytical viewpoint.
3. 3D Instance Segmentation with Multi-Object Assignment algorithm: Distinguish and analyze individual bundles with striking clarity.
4. Delete bundles: delete unwnated bundles on the boundaries 
6. Regional Bundle Categorization: Classify bundles distinctly as BASE, MIDDLE, or APEX regions.
7. IHC vs. OHC Identification: Accurately differentiate between Inner Hair Cells and Outer Hair Cells.
8. Comprehensive Measurement Computation: Determine volumes, centroids, surface areas, and more with precision.
9. Protein Intensity Quantification: Analyze the EPS8 protein with unrivaled accuracy.
10. 3D Distance Mapping: Chart the distance from peak to base for each bundle in three dimensions, tailored to your sample's resolution.

VASCilia stands as a treasure trove for the ear research community, heralding an era of simplicity and comprehensive measurements that resonate with the needs of laboratories worldwide. Alongside the plugin,  present a full-fledged collection of models for seamless 3D segmentation and region classification.

Moreover, we champion the advancement of knowledge by offering a dedicated training section. Whether you're delving into alternate staining methods or yearning to annotate, refine, and train new segmentation models, VASCilia empowers researchers to reach new heights in cochlear analysis.

Join the vanguard of auditory research with VASCilia—where sophistication meets simplicity.

![Pipeline Diagram](images/VASCilia.png)


# How to use PhenoRapid

There are four main sections:

**Section 1: User Configuration:** you need to setup the project path in the code, add the samples that you want to analyze and prepare the marker info tables assotiated to them.   
![Section1: User Configuration](images/User_config.png)

**Section 2: Analysis of Individual Sampels:** The algorithm starts to do the analysis starting by stardist segmentation, membrane segmentation, create cell information tables, calculate average intensities, run clustering, and producing all the tables and plots.

![Section2: Analysis of Individual Sampels](images/Analysis_of_Individual_Sampels.png)

**Section 3: Analysis after User Phenotyping** At this stage the user needs to phenotype and re-run this section to get all the plots and tables after phenotyping.

![Section3: Analysis after phenotype](images/Analysis_after_phenotype.png)

**Section 4: Analysis of Combined Samples:** Run this section if you want to get the results from multiple samples. 

![Section4: Analysis_of_combined_samples](images/Analysis_of_combined_samples.png)


## How to install :  

Open your linux terminal, install anaconda and copy and paste this command:

conda create -n rapids-22.04 -c rapidsai -c nvidia -c conda-forge \
    rapids=22.04 python=3.8 cudatoolkit=11.5 dask-sql

conda activate rapids-22.04

pip install scanpy

### Project Collaborators and Contact

**Author:** Dr. Yasmin M. Kassim    
**Contact:** ymkgz8@mail.missouri.edu

Contact me if you are intersted about other machine learning and computer vision tasks such as augmentation, dataset annotation, preprocessing, 2D/3D segmentation, 2D/3D object detection, classsification, and denoising.
