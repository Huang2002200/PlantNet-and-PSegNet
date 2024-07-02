# PlantNet and PSegNet<br>
This repos contains the official codes for papers:<br>
<br>
**PlantNet: A dual-function point cloud segmentation network for multiple plant species**<br>
[D. Li†](https://davidleepp.github.io/), G. Shi†, J. Li, Y. Chen, S. Zhang, S. Xiang, and S. Jin*<br>
<ins>†</ins> *Equal contribution*<br>
Pubilshed on *ISPRS Journal of Photogrammetry and Remote Sensing* in 2022<br>
[[Paper](https://www.sciencedirect.com/science/article/pii/S0924271622000119)]<br>
<br>
**PSegNet: simultaneous semantic and instance segmentation for point clouds of plants**<br>
[D. Li](https://davidleepp.github.io/), J. Li, S. Xiang, and A. Pan*<br>
Published on *Plant Phenomics* in 2022<br>
[[Paper](https://spj.science.org/doi/full/10.34133/2022/9787643?adobe_mc=MCMID%3D14000805405683999525849378418609464876%7CMCORGID%3D242B6472541199F70A4C98A6%2540AdobeOrg%7CTS%3D1700524800)]
***
## Prerequisites<br>
The code has a tensorflow version and a pytorch version, and their corresponding configurations are as follows:
* All deep networks run under Ubuntu 20.04
* Tensorflow version:<br>
  * Python == 3.7.13
  * Tensorflow == 1.13.1
  * CUDA == 11.7
* Pytorch version:
  * Python == 3.8.18
  * Pytorch == 2.0.1
  * CUDA == 12.4
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
***<p align="center">Pipeline of the PlantNet framework***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet/assets/64185853/9b0c7e00-11c6-4930-9baa-a7395719a955" width="60%" height="60%">
<br><br>
***<p align="center">Architecture of PlantNet. (a) is the main structure of the network, (b) is a clear demonstration of the Local Feature Extraction Operation (LFEO) used multiple times in the encoder***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet/assets/64185853/0be64f82-f628-45fb-a50d-aecf251ca468" width="90%" height="90%"><br><br>
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
***<p align="center">Schematic diagram of the VFPS downsampling strategy***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet/assets/64185853/34fbbb59-7ef1-4af2-90d1-3dc2192e9934" width="50%" height="50%"><br><br>
***<p align="center">Architecture of PSegNet***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet/assets/64185853/2d87fa66-d8d8-4662-a153-fcf706123de9" width="90%" height="90%"><br><br>
***<p align="center">Demonstration of DNFEB. In this figure, we only
display how features are processed by the 4th DNFEB in PSegNet***<br><br>
<img src="https://github.com/Huang2002200/PlantNet-and-PSegNet/assets/64185853/a76d84ac-c00d-4441-abeb-14bc910b2105" width="90%" height="90%"><br><br>
## Quick Start<br>
This project contains four main folders<br>
folder [**Original_Dataset**] contains the raw plant 3D data used in the paper, and the dataset is represented in txt files<br>
folder [**Data_preprocessing**] contains the code for transfering the raw dataset into the h5 format for network training and testing<br>
folder [**PlantNet**] contains the TensorFlow and Pytorch code of PlantNet<br>
folder [**PSegNet**]  contains the TensorFlow and Pytorch code of PSegNet<br>
### Original_Dataset<br>
Dataset includes 546 single-plant point clouds of three types of crops (tobacco, tomato, and sorghum) under 3 to 5 different growth environments (ambient light, 
shade, high heat, high light, drought) during a 20-day growth process.Of these, 105 point clouds for tomato, 312 point clouds for tobacco, and 129 point clouds for sorghum.<br>
The raw pointclouds are all txt files, each txt file contains 6 columns, the first three columns are xyz location information,
the fourth column is instance labeling, the fifth column is semantic labeling, and the sixth column is object labeling.<br>
### Data_preprocess<br>
Raw data needs to be preprocessed before it can be fed into the network for training or testing.<br>
According to different downsampling strategies (vanilla FPS,3DEPS,VFPS), we provide 3 preprocessing flow folders.
* floder [**FPS**] corresponds to use vanilla FPS in the preprocessing session.<br>
  * **000split test set and training set.py** is used to divide the pointclouds into a training set and a test set.<br>
  * **001data augmentation by FPS.py** is used to augment the test set and training set separately using the FPS approach.<br>
  * **002TXT2H5.py** is used to add semantic labels to pointclouds as well as convert the files from txt format to h5 format.<br>
* floder [**3DEPS**] corresponds to use 3DEPS in the preprocessing session.<br>
  * **000Batch differentiate and save point cloud edge and center points(c++).cpp** is used to separate plant point clouds into edge points and non-edge points (in batches), respectively.<br>
  * **001Merge edge and core points(4096+4096).py** is used to randomly sample 4096 points at the edge points and 4096 points at the center points,and merge the collected points into one point cloud file.<br>
  * **002split test set and training set.py** is used to divide the pointclouds into a training set and a test set.<br>
  * **003Proportionally merge into a new point cloud while expanding by a factor of 10.py** is used to sample 4096*(ratio) points at the edges and 4096*(1-radio) points at the center via vanilla FPS, which are then merged into a single point cloud.When merging, the FPS automatically carries out 10 times data augmentation.<br>
  * **004TXT2H5.py** is used to add semantic labels to pointclouds as well as convert the files from txt format to h5 format.<br>
* floder [**VFPS**] corresponds to use VFPS in the preprocessing session.<br>
  * **000Voxel downsampling.py** is used to sample pointclouds via voxel sampling method.<br>
  * **001split test set and training set.py** is used to divide the pointclouds into a training set and a test set.<br>
  * **002data augument FPS_batch.py** is used to performs vanilla FPS of point cloud files with inconsistent voxel downsampling points, while realizing data enhancement.<br>
  * **003TXT2H5.py** is used to add semantic labels to pointclouds as well as convert the files from txt format to h5 format.<br>
#### Notice! Programs need to be run one by one in the order of name and number.<br>
### PlantNet<br>
Contains all the code for training PlantNet networks in pytorch environment as well as in tensorflow environment.<br>
* floder [**models**] contains the code for PlantNet's entire training and testing process.<br>
  * **00estimate_mean_ins_size.py** is used to predict the approximate volume of instances in the training set, and the mean_ins_size.txt file is generated to assist subsequent clustering.<br>
  * **01train.py** is used to train the model parameters using the training set.<br>
  * **02test.py** is used to get the predicted labels by testing on the test set using the saved model parameters.<br>
  * **03eval_iou_accuracy.py** is used to compute quantitative metrics for instance segmentation as well as semantic segmentation.<br>
  * **04changeresulttoins.gt,sem.gt,ins,sem.py** is used to output pointclouds based on the predicted labels, which facilitates visual qualitative comparison.<br>
  * **model.py** contains the PlanetNet model and the loss function.
### PSegNet<br>
Contains all the code for training PSegNet networks in pytorch environment as well as in tensorflow environment.<br>
The code architecture is very similar to the one in the Plantnet folder.
## Citation<br>
Please consider citing our papers if the project helps your research with the following BibTex:
```
@article{li2022plantnet,
  title={PlantNet: A dual-function point cloud segmentation network for multiple plant species},
  author={Li, Dawei and Shi, Guoliang and Li, Jinsheng and Chen, Yingliang and Zhang, Songyin and Xiang, Shiyu and Jin, Shichao},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={184},
  pages={243--263},
  year={2022},
  publisher={Elsevier}
}
```
```
@article{li2022psegnet,
  title={PSegNet: Simultaneous semantic and instance segmentation for point clouds of plants},
  author={Li, Dawei and Li, Jinsheng and Xiang, Shiyu and Pan, Anqi},
  journal={Plant Phenomics},
  year={2022},
  publisher={AAAS}
}
```





