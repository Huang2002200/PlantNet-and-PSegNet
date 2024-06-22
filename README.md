# PlantNet and PSegNet<br>
This repos contains the official codes for papers:<br>
<br>
**PlantNet: A dual-function point cloud segmentation network for multiple plant species**<br>
[D. Li†](https://davidleepp.github.io/), G. Shi†, J. Li, Y. Chen, S. Zhang, S. Xiang, and S. Jin<br>
<ins>†</ins> *Equal contribution*<br>
Pubilshed on *ISPRS Journal of Photogrammetry and Remote Sensing* in 2022<br>
[[Paper](https://www.sciencedirect.com/science/article/pii/S0924271622000119)]<br>
<br>
**PSegNet: simultaneous semantic and instance segmentation for point clouds of plants**<br>
[D. Li](https://davidleepp.github.io/), J. Li, S. Xiang, and A. Pan<br>
Published on *Plant Phenomics* in 2022<br>
[[Paper](https://spj.science.org/doi/full/10.34133/2022/9787643?adobe_mc=MCMID%3D14000805405683999525849378418609464876%7CMCORGID%3D242B6472541199F70A4C98A6%2540AdobeOrg%7CTS%3D1700524800)]
***
## Ackonwledgement<br>
***
## Introduction<br>
### PlantNet<br>
The accurate plant organ segmentation is crucial and challenging to the quantification of plant architecture and 
selection of plant ideotype. The popularity of point cloud data and deep learning methods make plant organ 
segmentation a feasible and cutting-edge research. However, current plant organ segmentation methods are 
specially designed for only one species or variety, and they rarely perform semantic segmentation (stems and 
leaves) and instance segmentation (individual leaf) simultaneously. <br>
<br>
This study innovates a dual-function deep learning neural network (PlantNet) to realize semantic segmentation and 
instance segmentation of two dicotyledons and one monocotyledon from point clouds. The innovations of the PlantNet include a 3D EdgePreserving Sampling (3DEPS) strategy for preprocessing input points, a Local Feature Extraction Operation 
(LFEO) module based on dynamic graph convolutions, and a semantic-instance Feature Fusion Module (FFM).<br>
***<p align="center">Overview of the proposed method***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet-Code/blob/main/images/PlantNet01.png" width="60%" height="60%"><br><br>
***<p align="center">Overview of PlantNet. (a) is the main structure of the network, (b) is a clear demonstration of the Local Feature Extraction Operation (LFEO) used in the encoder***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet-Code/blob/main/images/PlantNet02.png" width="90%" height="90%"><br><br>
### PSegNet<br>
Phenotyping of plant growth improves the understanding of complex genetic traits and eventually expedites the development of
modern breeding and intelligent agriculture. In phenotyping, segmentation of 3D point clouds of plant organs such as leaves and
stems contributes to automatic growth monitoring and reflects the extent of stress received by the plant.<br><br>
In this work, we first
proposed the Voxelized Farthest Point Sampling (VFPS), a novel point cloud downsampling strategy, to prepare our plant
dataset for training of deep neural networks. Then, a deep learning network—PSegNet, was specially designed for segmenting
point clouds of several species of plants. The effectiveness of PSegNet originates from three new modules including the
Double-Neighborhood Feature Extraction Block (DNFEB), the Double-Granularity Feature Fusion Module (DGFFM), and the
Attention Module (AM).<br>
***<p align="center">Schematic diagram of the VFPS strategy***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet-Code/blob/main/images/PSegNet01.png" width="50%" height="50%"><br><br>
***<p align="center">The architecture of PSegNet. DNFEB is feature extraction block to condense the features***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet-Code/blob/main/images/PSegNet02.png" width="90%" height="90%"><br><br>
***<p align="center">Demonstration of DNFEB.In this figure, we only
display feature dimensions of the 4th DNFEB***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet-Code/blob/main/images/PSegNet03.png" width="90%" height="90%"><br><br>
## Quick Start<br>
This project contains four main folders<br>
floder [**dataset**] contains the raw dataset used in the paper as well as the preprocessed training h5 file and test h5 file<br>
floder [**data_preprocess**] contains the code for processing the raw dataset into the desired h5 format for the network<br>
floder [**PlantNet**] contains the code corresponding to the PlantNet paper<br>
floder [**PSegNet**]  contains the code corresponding to the PSegNet paper<br>
### dataset<br>
Dataset includes 558 single-plant point clouds of three types of crops (tobacco, tomato, and sorghum) under 3 to 5 different growth environments (ambient light, 
shade, high heat, high light, drought) during a 20-day growth process.Of these, 105 point clouds for tomato, 312 point clouds for tobacco, and 129 point clouds for sorghum.<br>
### data_preprocess<br>
Raw data needs to be preprocessed before it can be fed into the network for training or testing.<br>
* **001modifify label foemat (python).py** is used to modify the label format of input batches; standardizes the data format for subsequent processing.<br>
* **002PCD2TXT(python).py** is used to convert files from PCD format to the txt format, which can be skipped when the input is already in txt format.<br>
* **003remove backgroud spots and noise.py** is used to remove background points and noise from the raw plant pointclouds.<br>
* **004add_object_label ins_label_minus_2.py** is used to add category labels to the pointclouds and subtract 2 from all the instance labels so that the minimum value is 0 instead of 2.<br>
* **005split test set and training set.py** is used to divide the pointclouds into a train set and a test set.<br>
* **006data augmentation by FPS.py** is used to augment the test set and train set separately using the FPS approach.<br>
* **007TXT2H5.py** is used to add semantic labels to pointclouds as well as convert the files from txt format to h5 format.<br>
#### Notice! Programs need to be run one by one in the order of name and number.<br>






