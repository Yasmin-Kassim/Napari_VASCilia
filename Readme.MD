# VASCilia: A Revolutionary Napari Plugin for Cochlear StereoCilia Analysis
 
 Embark on a groundbreaking journey through the intricacies of the cochlea with VASCilia, the pioneering Napari plugin designed for the 3D segmentation and quantification of stereocilia bundles. Unveiling a suite of powerful features, VASCilia is a beacon of innovation for the auditory research realm, offering:

1. Selective Slice Discovery: Traverse the 3D landscape to pinpoint pivotal slices within your stack.
2. Stack Rotation: Align your perspective with precision for an optimal analytical viewpoint.
3. 3D Instance Segmentation with Multi-Object Assignment algorithm: Distinguish and analyze individual bundles with striking clarity.
4. Delete bundles: delete unwnated bundles.
6. Regional Bundle Categorization: Classify bundles distinctly as BASE, MIDDLE, or APEX regions.
7. IHC vs. OHC Identification: Accurately differentiate between Inner Hair Cells and Outer Hair Cells.
8. Comprehensive Measurement Computation: Determine volumes, centroids, surface areas, and more with precision.
9. Protein Intensity Quantification: Analyze the EPS8 protein with unrivaled accuracy.
10. 3D Distance Mapping: Chart the distance from peak to base for each bundle in three dimensions, tailored to your sample's resolution.

VASCilia stands as a treasure trove for the ear research community, heralding an era of simplicity and comprehensive measurements that resonate with the needs of laboratories worldwide. Alongside the plugin,  present a full-fledged collection of trained models for seamless 3D segmentation and region classification.

Moreover, we champion the advancement of knowledge by offering a dedicated training section. Whether you're delving into alternate staining methods or yearning to annotate, refine, and train new segmentation models.

Join the vanguard of auditory research with VASCilia—where sophistication meets simplicity.

![Pipeline Diagram](images/VASCilia.png)


## How to install :  

Open your linux terminal, install anaconda and copy and paste this command:

conda create -n rapids-22.04 -c rapidsai -c nvidia -c conda-forge \
    rapids=22.04 python=3.8 cudatoolkit=11.5 dask-sql

conda activate rapids-22.04

pip install scanpy

### Project Authors and Contacts

**Python Implementation of this repository:** Dr. Yasmin M. Kassim    
**Contact:** ykassim@ucsd.edu

**Stacks used in this study imaged by:** Dr. David Rosenberg   
**Contact:** d2rosenberg@UCSD.EDU

Four undergraduate students participated in annotating the manual ground truth using CVAT annotation tool: Zhuoling Huang, Samia Rahman, Ibraheem Al Shammaa, Samer Salim, and Kevin Huang 

**Lab Mentor:** Dr. Uri Manor   
**Contact:** u1manor@UCSD.EDU




