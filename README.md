# Applied Deep Learning

This is a personal repository for the coursework assesment of the master-level unit in Applied Deep Learning (COMSM0018 - 2019/2020), taken at the University of Bristol for the Master of Engineering (MEng) degree.  

The coursework assesses skill in applying deep learning knowledge by replicating a published research on a SOTA model for sound classification. The referenced work is '_Environment Sound Classification using a Two-Stream CNN Based on Decision-Level Fusion_' by Yu Su et al. You can review the paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/pdf/sensors-19-01733.pdf) or read the attached [pdf file](Environmental_Sound_Classification.pdf). Additionally, our group report detailing the process of our implementation can be found in this repository [here](Applied_Deep_Learning_Coursework.pdf).

Our code was implemented using PyTorch version 1.2.0, and was written for high performance computing where training was run on the University of Bristol's supercomputer _BlueCrystal4_. Our result achived up to a 5% accuracy difference as those reported by the published work - given some adjustment from the coursework instruction. Overall, we achived a First-Class mark for our implementation.

### List of Dependencies
- Python 3.7
- PyTorch 1.2.0
- torchvision 0.4.0
- matplotlib 3.1.1
- seaborn 0.9.0
- scikit-learn 0.23.1
- tensorflow 1.14 (primarily for tensorboard, a metrics logging tool)

---
## Overview

### Introduction

Environmental Sound Classification (ESC) is a major task in the growing field of Intelligent Sound Recognition technology; which differ from other areas of sound recognition such as Speech Recognition or Music Recognition. Inherently, the acoustic features of environmental sounds differ to those of speeches or music, and as such comes the main challenges in ESC tasks: composing an appropriate input feature set and developing a well-performing model for ESC. So far existing models are still unsatisfactory, but, due to advent of deep learning models some progress has been made, and a notable example is the published work of Su et al, which claims state-of-the-art solution. In this report, we replicated such results using the proposed architecture, and see if further improvements can still be made.

Therefore, to solve both of those problems, Su et al claim that (1) combination of acoustic features used in speech or music recognition via late fusion methods would make a suitable features set for ESC. And that (2) a stacked neural network of two-streamed CNNs (with late-fusion during testing) can outperform other models of the time on known datasets such as the _UrbanSound8K_.

## The dataset and input features

The dataset used is the _UrbanSound8K_ dataset which consist of 10 classes of environmental sounds. The dataset is comprised of 8732 audio clips, each labelled as one of the 10 following classes: 
- **air conditioner** (0) 
- **car horn** (1)
- **children playing** (2), 
- **dog bark** (3)
- **drilling** (4)
- **engine idling** (5) 
- **gunshot** (6)
- **jackhammer** (7)
- **siren** (8)
- **street music** (9).

The train and test sets of the dataset are saved in *UrbanSound8K_train.pkl* and *UrbanSound8K_test.pkl*, both can be found in the [code](/code) folder.

The dataset is structured as a list of dictionaries and in the format of PyTorch abstract data class. Each dict in the list corresponds to a different audio segment from an audio file. The dicts contain the following keys:

- **filename**: contains a unique name of the audio file. This is useful for matching audio segments to the audio file that they are coming from, and compute global scores by averaging the segments scores that have the same filename
-	**class**: class name
-	**classID**: class number  [0…9]
-	**features**: all the acoustic features to be used for training:
    - *logmelspec* (LM)
    - *mfcc* (MFCC)
    - *chroma*
    - *spectral_contrast*
    - *Tonnetz*

Below is a spectogram representation of the 5 acoustic input features. Note, the last 3 features (*chroma*.*spectral_contrast*,*Tonnetz*) are aggregated to form one feature set called CST.

<!-- ![Inputs](/figures/input_features.png)  -->
<centre><img src="/figures/input_features.png" alt="input" width="600"/></centre>

Furthermore, as part of the ETL process, we used PyTorch's DataLoader utility to load the different inputs for training the network.

## The model architecture

The main architecture of the model proposed is a stacked convolutional neural network (CNN) of two-stream identical network with 4 convolution layers, and 2 fully connected layers with a softmax layer at the end. 

The top stream uses the concatentaion of the LM input feature and CST; dubbed *LMCNet*. Whilst the bottom stream, uses MFCC and CST as its inputs and is dubbed *MCNet*.

The main architecture of the paper would be denoted as *TCSNNNet*, in which both the *LMCNet* and *MCNet* streams were trained independently, and their predictions combined during testing i.e. late-fusion. Additionally, to confirm late-fusion effectiveness, another variation of the
architecture, denoted as *MLMCNet* is created. Where all five features are combined linearly into one feature set named MLMC, and then passed into a single CNN identical to either of the streams. The crux of the original authors’ claim is that predictions made with late-fusion (*TCSNNNet*) would outperform other models such as *MLMCNet*, *LMCNet*, *MCNet*.

Below is a personally illustrated depiction of the architecture.

![Architecture](/figures/architecture.png)

Top stream is LMCNet, bottom stream is MCNet. Yellow layers indicate convolution layers, whilst purple ones are fully-connected networks. The red layer represents a pooling layer after max-pooling, to add clarity to the confusion in the paper. Filters of size 3x3xd are convolution filters, whilst 2x2xd are pooling filters, and d being 32 or 64 is the hyperparameter indicating the number of filters used in that specific layer.

## Our Result

Overall, our own attempt at replicating their work shows favourable and consistent results in support of their claim. The performance of *TSCNNNet* achieves the highest accuracy out of other tested models; indicating that late fusion using the proposed architecture does perform better than input level fusion (like in *MLMCNet*) in terms of accuracy. See our group report [here](Applied_Deep_Learning_Coursework.pdf) for further details.




