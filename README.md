# Lung-Cancer-Diagnosis
Lung nodules  are potential  manifestations of  lung cancer, and their  early detection facilitates  early treatment and improves  patient’s  chances  for  survival.This project proposes a method which tries to improve on the lung cancer detection system by proper segmentation of lung nodules on different slices of the CT scans and then tries to apply deep learning methodology like Convolution Neural Networks (CNN) using TensorFlow framework on those segmented scan slices and discards the unnecessary information in order to narrow down the relevant slices and predict whether the patient has lung cancer in the final layer of the fully connected network.Data Preprocessing and Data Augmentation constitute two most important steps in the Image preprocessing.

## Data Description
The primary dataset is the Lung Image Database Consor-tium image collection (LIDC-IDRI) [LIDC dataset](https://wiki.cancerimagingarchive.net) consists of diagnos-tic and lung cancer screening thoracic computed tomogra-phy (CT) scans with marked-up annotated lesions. We've preprocessed this data and stored them in a pickle file, which you can [download here](https://drive.google.com/file/d/1Wn7RqGkiq3lanlRCKlLU7U_mcGZUdwly/view?usp=sharing). After downloading the files you should place them in a folder called 'data'. After that, you can train the UNet on the LIDC dataset using the provided in train.py file.

## The model architecture
We plan to use the pretext part of the state-of-the-art SimCLR model in order to get a better visual representation of our data, and then parse it into the U-Net model for the downstream task of segmentation/classification of lung nodules from CT scans.
![](https://github.com/makama-md/lungD_Project/blob/main/plots/uu.png)

## download trained model
The trained model can be downloaded from [download here](https://drive.google.com/file/d/10F7U-8ZjRWAHvCJKZEtR4XnQkI9tyyY-/view?usp=sharing)

* train loss, train accuracy and precision

<p align="center">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/training%20loss.png" width="350" title="hover text">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/training%20accuracy.png" width="350" alt="accessibility text">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/precision.png" width="350" alt="accessibility text">
</p>



## Result
* Nodule Segmentation 

![](https://github.com/makama-md/lungD_Project/blob/main/result/segmentated%20result.png)



