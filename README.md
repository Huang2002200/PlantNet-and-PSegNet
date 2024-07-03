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
folder [**Data_preprocessing**] contains the code for converting the raw dataset into the h5 format for network training and testing<br>
folder [**PlantNet**] contains the TensorFlow and Pytorch code of PlantNet<br>
folder [**PSegNet**]  contains the TensorFlow and Pytorch code of PSegNet<br>

### Original_Dataset<br>
The dataset includes 546 single-plant point clouds of three types of crops (tobacco, tomato, and sorghum) under 3 to 5 different growth environments (ambient light, 
shade, high heat, high light, drought) during a 20-day growth period. In the total dataset, 105 point clouds are for tomato, 312 point clouds are for tobacco, and 129 point clouds are for sorghum.<br><br>
  The raw point clouds are all represented in "txt" files. Each single txt file is a single 3D plant. Each row of the txt file stands for a point in that point cloud. Each txt file contains 6 columns, in which the first three shows the "xyz" spatial information,
the fourth column is the instance label, the fifth column is the semantic label, and the sixth column is the object label (nevern used in our project).<br><br>
  The value of semantic labels starts at "0" and ends at "5". Each semantic label number means a sepecies-relavant crop organ; e.g., "0" means "the stem system of tobacco", and "1" means "the leaf of tobacco". The value of instance label, in most cases, stands for the label of each leaf organ instance; e.g., "1" means the 1st leaf of the current point cloud, and "18" means the 18th leaf of the current point cloud. It should be noted that the instance label is not consecutive, which means "1" is not followed by "2", but may be "5". It should also be noted that the stem system only has one instance--itself, because one cannot divide biologically meaningful stem instances from the total stem system of a crop. The last thing about labels is that the instance label of every point from the stem system (no matter which species) is assigned 0.<br>

### Data_preprocessing<br>
Raw data needs to be preprocessed before it can be fed into networks for training or testing.<br>
We provide 3 different preprocessing techniques (with different downsampling strategies) to prepare the data for networks.<br>
* folder [**FPS**] refers to using Farthest Point Sampling (FPS) in the preprocessing session.<br>
  * file **000split test set and training set.py** is used to randomly divide the point clouds into a training set and a testing set.<br>
  * file **001data augmentation by FPS.py** is used to downsample and augment (default 10x) the testing set and the training set separately using FPS.<br>
  * file **002TXT2H5.py** is used to convert the txt files into h5 format packages. Both versions of PlantNet or PSegNet accept h5 files as input.<br>
* folder [**3DEPS**] refers to using 3D Edge-Preserving Sampling (3DEPS) in the preprocessing session.<br>
  * file **000Batch differentiate and save point cloud edge and center points(c++).cpp** is used to separate original point clouds into point clouds containing only edge points and point clouds containing only non-edge points (in batches), respectively.<br>
  * file **001Merge edge and core points(4096+4096).py** uses FPS to randomly sample 4096 points from the edge part and sample 4096 points from the non-edge part, and combine the two parts into a new point cloud with 4096+4096 points.<br>
  * file **002split test set and training set.py** is used to divide the new point clouds into a training set and a testing set.<br>
  * file **003Proportionally merge into a new point cloud while expanding by a factor of 10.py** uses FPS to sample 4096*(ratio) points from the edge part and to sample 4096*(1-radio) points from the non-edge part, then merges the two parts into a new single point cloud with 4096 points. When merging, the FPS is automatically carried out 10 times to do 10x data augmentation. If you want to create more data, please change the factor "10" to a larger number.<br>
  * file **004TXT2H5.py** is used to convert the files from txt format into h5 format for network input.<br>
* folder [**VFPS**] refers to using Voxelized Farthest Point Sampling (VFPS) in the preprocessing session.<br>
  * file **000Voxel downsampling.py** is used to sample point clouds via voxel sampling method. Please be noted that in order to acquire desirable results, it is suggested to tune the voxel size parameter to allow the number of voxels to be around 10000+. The number of voxels must be larger than the final point number (such as 4096 in our project).<br>
  * file **001split test set and training set.py** is used to divide the voxeled point clouds into a training set and a testing set.<br>
  * file **002data augument FPS_batch.py** performs FPS on the voxelized point cloud files to control the number of points to a fixed value such as 4096. We also realize 10 times data augmentation by randomly initialize the first point of FPS for 10 times. The times of data augmentation can be controlled by the user.<br>
  * file **003TXT2H5.py** is used to convert the files from txt format to the h5 format for further training and testing.<br>
#### Notice! files need to be run one by one in the order of names. The PlantNet and PSegNet can accept all three types of inputs generated by the above preprocessing techniques, and the performances have only little difference. <br><br>
### PlantNet<br>
Contains all code for training PlantNet networks in both pytorch environment and TensorFlow environment.<br>
* folder [**models**] contains the code for PlantNet's entire training and testing processes.<br>
  * file **00estimate_mean_ins_size.py** is used to predict an approximated volume of organ instances in the training set, and the "mean_ins_size.txt" file is generated to assist subsequent clustering.<br>
  * file **01train.py** is used to train the model parameters using the training set.<br>
  * file **02test.py** is used to get the predicted labels by testing on the test set using the saved model parameters.<br>
  * file **03eval_iou_accuracy.py** is used to compute the quantitative metrics for the organ instance segmentation task as well as the organ semantic segmentation task.<br>
  * file **04changeresulttoins.gt,sem.gt,ins,sem.py** is used to output segmented point clouds based on the predicted labels, which facilitates the visual qualitative comparison with ground truth.<br>
  * file **model.py** contains the full PlantNet model and the loss function.<br>
  * we also provide a trained PyTorch PlantNet model at epoch 198; the model parameters were saved into a file "PlantNet/PlantNet_pytorch/models/checkpoints/model_epoch198.pth".<br>
### PSegNet<br>
Contains all code for training PSegNet networks in both pytorch environment and TensorFlow environment.<br>
The folder architecture is very similar to the one in the PlantNet folder therefore we omit the instructions.<br><br>
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





