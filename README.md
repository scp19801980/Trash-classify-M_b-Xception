Trash-classify: M_b-Xception
====
We propose an effective method to improve the network performance on TrashNet data set. This method widens the network by expanding branches, and then uses add layers to realize the fusion of feature information. Using this method to replace the core structure of the Xception network, a network structure that provides high precision on the TrashNet dataset is obtained. We call it Multi-branch Xception (M-b Xception).

Models
==========
There are three model files.  
model1: Xception  
model2: M_b Xception. The number of convolution channels of core structure is 728.  
model3: M_b Xception. The number of convolution channels of core structure is 896.  

Data set
======
The data set is widely used in the field of garbage classification based on deep learning technology. There are 2527 RGB images, including 501 glass images, 594 paper images, 403 cardboard images, 482 plastic images, 410 metal images and 137 trash images.   
You can use this link to download the original data set：https://pan.baidu.com/s/13XTKEDKRlUtq7Ao45tj2CA   password ：Mbxc

Environment
========
Python 3.5.2

cuda 8.0

cudnn 6.0.21

tensorflow-gpu 1.4.0

keras 2.1.3

numpy 1.16.2



If this project is helpful to you, please cite the following paper:

C. Shi, R. Xia and L. Wang, "A Novel Multi-Branch Channel Expansion Network for Garbage Image Classification," in IEEE Access, doi: 10.1109/ACCESS.2020.3016116.


