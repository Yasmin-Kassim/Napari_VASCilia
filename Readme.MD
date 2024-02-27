# Vision Analysis Tool for StereoCilia Hair Bundles in the Cochlea
 
 We created VASCilia is the first Napari plugin to 3D segment and quantify the stereocilia bundles in the cochlea. It has many functions including 
1) finding the slices of interest within the 3D stack, 
2) Stack Rotation, 
3) 3D Instance Segmentation, 
4) Multi-object assignment for the bundles, 
5) different ways of visualization, 
6) Region classification (BASE, MIDDLE, APEX), 
7) Identify IHCs vs OHCs, 
8) Calculate different measurments (volume, centroid, surface area, ..etc), 
9) Compute protein intensity (EPS8 protein), 10) Find the distance from the peak to the base of each bundle in 3D based on sample resolution.

VASCilia plugin will be a fortune for ear research community for it's use simplicity and the many measurments that are widely needed by almost every lab works in this area.
We provide all the models that is needed to perform 3D segmentation and also region classification.
We also make a full training section available to the research community for their datasets if they use different staining. Through the training section, you can annotate, re-fine predicted objects and train a new 3D segmentation model.

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
