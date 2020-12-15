# Lung-Cancer-Diagnosis
Lung nodules  are potential  manifestations of  lung cancer, and their  early detection facilitates  early treatment and improves  patient’s  chances  for  survival.This project proposes a method which tries to improve on the lung cancer detection system by proper segmentation of lung nodules on different slices of the CT scans and then tries to apply deep learning methodology like Convolution Neural Networks (CNN) using TensorFlow framework on those segmented scan slices; and discards the unnecessary information in order to narrow down the relevant slices, and predict whether the patient has lung cancer in the final layer of the fully connected network.

## Data Description
The primary dataset used is the Lung Image Database Consortium image collection (LIDC-IDRI) [LIDC dataset].(https://wiki.cancerimagingarchive.net) consists of diagnostic and lung cancer screening computed tomography (CT) scans with marked-up annotated lesions. We have preprocessed this data and stored them as a pickle file, which you can [download here](https://drive.google.com/file/d/1Wn7RqGkiq3lanlRCKlLU7U_mcGZUdwly/view?usp=sharing). After downloading the files you should place them in a folder called 'data'.

## The Model Architecture
We used the state-of-the-art segmentation model architecture, UNET. We used some of data augmentation strategy used in the state-of-the-art SimCLR model in order to get a better visual representation of our data, before parsing it into the U-Net model for the downstream task of segmentation/classification of lung nodules from CT scans.Data Preprocessing and Data Augmentation constitute two most important steps in the Image preprocessing.
![](https://github.com/makama-md/lungD_Project/blob/main/plots/uu.png)

## Training
In order to train the model, you should first download the already preprocessed data from [download here](https://drive.google.com/file/d/1Wn7RqGkiq3lanlRCKlLU7U_mcGZUdwly/view?usp=sharing). and put it a folder called 'data'. Then you can call the dataset.py and data_aug.py files to create training, validation and test data and applying data augmentation before calling the train.py file to train the model. You can then make prediction on the test data by calling the predict.py file.

### Download the Pre-trained Model
You can as well make inference on the test data by downloading the pretrained model from [download here](https://drive.google.com/file/d/10F7U-8ZjRWAHvCJKZEtR4XnQkI9tyyY-/view?usp=sharing)

### Training loss, Training accuracy and Precision of the Model

<p align="center">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/training%20loss.png" width="350" title="hover text">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/training%20accuracy.png" width="350" alt="accessibility text">
  <img src="https://github.com/makama-md/lungD_Project/blob/main/plots/precision.png" width="350" alt="accessibility text">
</p>

## Result
<img src="https://github.com/makama-md/lungD_Project/blob/main/plots/model_performance.png" width="550" alt="accessibility text">

* Samples of Nodules detected by the model

![](https://github.com/makama-md/lungD_Project/blob/main/result/segmentated%20result.png)



