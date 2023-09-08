# Subsampled-MRI-Reconstruction-Techniques

This repo's aim is to describe the coding work of my Master Thesis in Biomedical Engineering. The initial purpose was to recreate and use the GRAPPA and SENSE algorythms to the Raw MRI data from UMCG MRI Scanners. 

The first file to look into is the Demo.ipynb, this file has the example of the procedure used for one slice of one patient. In this file everything is justified and commented. Also some plots were made to better understand the data it was used. 

All funtions created are given in the reconstruction_functions.py file and explained there aswell.

grappaauto.py and senseauto.py is my usage of the code to loop over all slices and patients. If you plan to use that you need to change some things like file paths and csv file name where the Image Quality Metrics are going to be saved.

