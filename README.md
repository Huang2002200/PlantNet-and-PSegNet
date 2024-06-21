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



