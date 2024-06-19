# PlantNet-and-PSegNet-Code <br>
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
[[Paper](https://spj.science.org/doi/full/10.34133/2022/9787643?adobe_mc=MCMID%3D14000805405683999525849378418609464876%7CMCORGID%3D242B6472541199F70A4C98A6%2540AdobeOrg%7CTS%3D1700524800)]<br>
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
(LFEO) module based on dynamic graph convolutions, and a semantic-instance Feature Fusion Module (FFM). 


