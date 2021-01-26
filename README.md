# Applied Deep Learning

This is a personal repository for the coursework assesment of the master-level unit in Applied Deep Learning (COMSM0018), taken at the University of Bristol for the Master of Engineering (MEng) degree.  

The coursework assesses skill in applying deep learning knowledge by replicating a published research on a SOTA model for sound classification. The referenced work is '_Environment Sound Classification using a Two-Stream CNN Based on Decision-Level Fusion_' by Yu Su et al. You can review the paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/pdf/sensors-19-01733.pdf) or read the attached [pdf file](Environmental_Sound_Classification.pdf). Additionally, our group report detailing the process of our implementation can be found in this repository [here](Applied_Deep_Learning_Coursework.pdf).

Our code was implemented using _PyTorch_ version 1.2.0 and was written for high performance computing. We trained the model on the University of Bristol's supercomputer _BlueCrystal4_. 

### List of Dependencies
- Python 3.7
- PyTorch 1.2.0
- torchvision 0.4.0
- matplotlib 3.1.1
- seaborn 0.9.0
- scikit-learn 0.23.1
- tensorflow 1.14 (primarily as it contains tensorboard, a metrics logging tool)

---
## Overview

### Introduction

Environmental Sound Classification (ESC) is a major task in the growing field of Intelligent Sound Recognition technology; which differ from other areas of sound recognition such as Speech Recognition or Music Recognition. Inherently, the acoustic features of environmental sounds differ to those of speeches or music, and as such comes the main challenges in ESC tasks: composing an appropriate input feature set and developing a well-performing model for ESC. So far existing models are still unsatisfactory, but, due to advent of deep learning models some progress has been made, and a notable example is the published work of Su et al, which claims state-of-the-art solution. In this report, we replicated such results using the proposed architecture, and see if further improvements can still be made.

The work done on Environmental Sound Classification (ESC) Using Two-streamed CNN with Decision Level Fusion by Su et al, is an attempt to produce a SOTA model for solving ESC tasks i.e. classifying sounds commonly heard in background noises. One of the challenges of ESC is that commonly known acoustic features, such as those typically used in speech or music recognition are insufficient representation of environmental sounds; since the latter are typically non-stationary and have higher temporal variance. Second known issue is an unsatisfactory history of classifiers for ESC. Existing models are either lacking in temporal invariance or failing to reach high accuracy prediction. 

Therefore, to solve both of those problems, Su et al claim that (1) combination of acoustic features used in speech or music recognition via late fusion methods would make a suitable features set for ESC. And that (2) a stacked neural network of two-streamed CNNs (with late-fusion during testing) can outperform other models of the time on known datasets such as the _UrbanSound8K_.


## The model architecture

The main architecture of the model proposed is a stacked CNN of two-stream identical network with 4 convolution layers, and 2 fully connected layers with a softmax layer at the end. Below is a personally illustrated depiction of the architecture.

![Architecture](/figures/architecture.png)

