# Applied Deep Learning

This is a personal repository for the coursework assesment of the master-level unit in Applied Deep Learning (COMSM0018), taken at the University of Bristol for the Master of Engineering (MEng) degree.  

The coursework assesses skill in applying deep learning knowledge by replicating a published research on a SOTA model. The referenced work is '_Environment
Sound Classification using a Two-Stream CNN Based on
Decision-Level Fusion_' by Yu Su et al. You can review the paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6479959/pdf/sensors-19-01733.pdf) or read the attached _Environmental_Sound_Classification_.pdf_ file. Additionally our group report detailing the process of our replication can be also be found in this repository [here](Applied_Deep_Learning_Coursework.pdf).


## Overview


Environmental Sound Classification (ESC) is a major task in the growing field of Intelligent Sound Recognition technology; which differ from other areas of sound recognition such as Speech Recognition or Music Recognition. Inherently, the acoustic features of environmental sounds differ to those of speeches or music, and as such comes the main challenges in ESC tasks: composing an appropriate input feature set and developing a well-performing model for ESC. So far existing models are still unsatisfactory, but, due to advent of deep learning models some progress has been made, and a notable example is the published work of Su et al, which claims state-of-the-art solution. In this report, we will attempt to replicate such results using the proposed architecture, and see if further improvements can still be made.

The work done on Environmental Sound Classification (ESC) Using Two-streamed CNN with Decision Level Fusion by Su et al, is an attempt to produce a state of the art model for solving ESC tasks i.e. classifying sounds commonly heard in background noises. One of the challenges of ESC is that commonly known acoustic features, such as those typically used in speech or music recognition are insufficient representation of environmental sounds; since the latter are typically non-stationary and have higher temporal variance. Second known issue is an unsatisfactory history of classifiers for ESC. Existing models are either lacking in temporal invariance or failing to reach high accuracy prediction. 

Therefore, to solve both of those problems, Su et al claim that (1) combination of acoustic features used in speech or music recognition via late fusion methods would make a suitable features set for ESC. And that (2) a stacked neural network of two-streamed CNNs (with late-fusion during testing) can outperform other models of the time on known datasets such as the _UrbanSound8K_.

![Architecture](/figures/architecture.png)